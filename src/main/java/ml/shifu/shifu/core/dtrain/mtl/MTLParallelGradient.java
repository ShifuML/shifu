/*
 * Copyright [2013-2019] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.mtl;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataSet;
import ml.shifu.shifu.util.CommonUtils;

/**
 * Gradients in {@link MTLWorker} are computed in parallel and accumulated in {@link MTLParallelGradient}.
 * 
 * <p>
 * {@link #threadNumber} is the parallel number. Training and validation data set are split according to such number.
 * Multiple threads would be submitted to run splits forward-backward computation in parallel. Finally accumulated
 * gradients are send out back to {@link MTLWorker} instance.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MTLParallelGradient {

    private static final Logger LOG = LoggerFactory.getLogger(MTLParallelGradient.class);

    /**
     * Parallel number used to do parallel training on gradient computation.
     */
    private int threadNumber;

    /**
     * The multi-task model instance used to compute gradients.
     */
    private MultiTaskModel mtm;

    /**
     * Training data set.
     */
    private FloatMLDataSet trainData;

    /**
     * Validation data set
     */
    private FloatMLDataSet validationData;

    /**
     * Parallel execution service instance.
     */
    private CompletionService<MTLParams> completionService;

    /**
     * Given {@link #threadNumber}, to identify training low and high indexes after splits.
     */
    private long[] trainLows;
    private long[] trainHighs;

    /**
     * Given {@link #threadNumber}, to identify validation low and high indexes after splits.
     */
    private long[] validationLows;
    private long[] validationHighs;

    /**
     * Multi-task model parallel gradient computation constructor.
     * 
     * @param mtm
     *            the multi-task model instance to compute gradients.
     * @param threadNumber
     *            number of threads or parallel
     * @param trainData
     *            the training data set split by {@link #threadNumber}
     * @param validationData
     *            the validation data set split by {@link #threadNumber}
     * @param completionService
     *            the {@link CompletionService} instance to schedule computing threads
     */
    public MTLParallelGradient(final MultiTaskModel mtm, int threadNumber, final FloatMLDataSet trainData,
            final FloatMLDataSet validationData, CompletionService<MTLParams> completionService) {
        this.threadNumber = threadNumber;
        this.mtm = mtm;
        this.trainData = trainData;
        this.validationData = validationData;
        this.completionService = completionService;

        assert threadNumber > 0 && threadNumber < 33;
        long recordCount = this.trainData.getRecordCount();
        this.trainLows = new long[threadNumber];
        this.trainHighs = new long[threadNumber];

        long stepCount = recordCount / threadNumber;
        if(recordCount % threadNumber != 0) {
            stepCount += (recordCount % threadNumber) / stepCount;
        }
        for(int i = 0; i < threadNumber; i++) {
            this.trainLows[i] = i * stepCount;
            if(i != threadNumber - 1) {
                this.trainHighs[i] = this.trainLows[i] + stepCount - 1;
            } else {
                this.trainHighs[i] = recordCount - 1;
            }
        }
        LOG.info("Train record count: {}", recordCount);
        LOG.info("Train lows: {}", Arrays.toString(trainLows));
        LOG.info("Train highs: {}", Arrays.toString(trainHighs));

        long testRecordCount = this.validationData.getRecordCount();
        this.validationLows = new long[threadNumber];
        this.validationHighs = new long[threadNumber];
        long testStepCount = testRecordCount / threadNumber;
        if(testRecordCount % threadNumber != 0) {
            // move step count to append last gap to avoid last thread worse 2*testStepCount-1
            testStepCount += (testRecordCount % threadNumber) / testStepCount;
        }
        for(int i = 0; i < threadNumber; i++) {
            this.validationLows[i] = i * testStepCount;
            if(i != threadNumber - 1) {
                this.validationHighs[i] = this.validationLows[i] + testStepCount - 1;
            } else {
                this.validationHighs[i] = testRecordCount - 1;
            }
        }

        LOG.info("Validation record count: {}", testRecordCount);
        LOG.info("Validation lows: {}", Arrays.toString(validationLows));
        LOG.info("Validation highs: {}", Arrays.toString(validationHighs));
    }

    /**
     * Submit gradient computation task in parallel and to accumulate all task results together after synced.
     * 
     * @return the accumulated gradients of such multi-task model instance.
     */
    public MTLParams doCompute() {
        long start = System.currentTimeMillis();
        for(int i = 0; i < this.threadNumber; i++) {
            this.completionService.submit(new GradientTask(this.mtm, this.trainData, this.validationData,
                    this.trainLows[i], this.trainHighs[i], this.validationLows[i], this.validationHighs[i]));
        }
        MTLParams params = null;
        for(int i = 0; i < this.threadNumber; i++) {
            try {
                // no need to take a timeout here as it is difficult to set a good default timeout for all cases.
                MTLParams paramsTmp = this.completionService.take().get();
                if(paramsTmp != null) {
                    if(params != null) {
                        params.combine(paramsTmp);
                    } else {
                        params = paramsTmp;
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (ExecutionException e) {
                LOG.error("ExecutionException exception", e);
            }
        }
        LOG.info("Worker with parallel train run time {} ms.", (System.currentTimeMillis() - start));
        return params;
    }

    /**
     * Runnable gradient computation task given split training data set and validation data set.
     */
    private static class GradientTask implements Callable<MTLParams> {
        private final Logger TASK_LOG = LoggerFactory.getLogger(GradientTask.class);

        /**
         * Copied multi-task model instance from {@link MTLParallelGradient}.
         */
        private MultiTaskModel mtm;
        private FloatMLDataSet trainData;
        private FloatMLDataSet validationData;
        private long trainLow;
        private long trainHigh;
        private long validationLow;
        private long validationHigh;

        public GradientTask(final MultiTaskModel mtm, final FloatMLDataSet trainData, final FloatMLDataSet testData,
                long trainLow, long trainHigh, long testLow, long testHigh) {
            this.mtm = mtm.clone();
            this.trainData = trainData;
            this.validationData = testData;
            this.trainLow = trainLow;
            this.trainHigh = trainHigh;
            this.validationLow = testLow;
            this.validationHigh = testHigh;
            if(this.trainData.getRecordCount() == 0) {
                throw new IllegalArgumentException("Empty training data set.");
            }
        }

        /**
         * Gradient computation logic based on split training data set and validation data set. Do forward-backward
         * propagation in training data set to compute gradients and then compute validation error based on current
         * model weights.
         */
        @Override
        public MTLParams call() throws Exception {
            if(this.mtm == null || validationHigh < validationLow || trainHigh < trainLow) {
                TASK_LOG.error(
                        "input parameters not correct, validationHigh={}, validationLow={}, trainHigh={}, trainLow={}",
                        validationHigh, validationLow, trainHigh, trainLow);
                throw new IllegalArgumentException(String.format(
                        "Input parameters not correct, validationHigh=%s, validationLow=%s, trainHigh==%s, trainLow==%s",
                        validationHigh, validationLow, trainHigh, trainLow));
            }

            long start = System.currentTimeMillis();
            // forward and backward compute gradients for each iteration
            double trainCnt = trainHigh - trainLow, validCnt = validationHigh - validationLow;
            double trainSize = 0, validationSize = 0;
            double trainSumError = 0d, validSumError = 0d;

            for(long i = trainLow; i < trainHigh; i++) {
                FloatMLDataPair data = BasicFloatMLDataPair.createPair(this.trainData.getInputSize(),
                        this.trainData.getIdealSize());
                this.trainData.getRecord(i, data);
                trainSize += data.getSignificance();
                double[] logits = this.mtm.forward(CommonUtils.floatToDouble(data.getInputArray()));
                double[] predict = CommonUtils.sigmoid(logits);
                double[] error = CommonUtils.minus(predict, data.getIdealArray());
                for(int j = 0; j < error.length; j++) {
                    trainSumError += (error[j] * error[j] * data.getSignificance());
                }
                this.mtm.backward(predict, CommonUtils.floatToDouble(data.getIdealArray()), data.getSignificance());
            }
            TASK_LOG.info("Worker with training time {} ms.", (System.currentTimeMillis() - start));

            start = System.currentTimeMillis();
            TASK_LOG.info("Start validation computation.");
            // compute validation error
            for(long i = validationLow; i < validationHigh; i++) {
                FloatMLDataPair data = BasicFloatMLDataPair.createPair(this.validationData.getInputSize(),
                        this.validationData.getIdealSize());
                this.validationData.getRecord(i, data);
                double[] logits = this.mtm.forward(CommonUtils.floatToDouble(data.getInputArray()));
                double[] predict = CommonUtils.sigmoid(logits);
                validationSize += data.getSignificance();
                double[] error = CommonUtils.minus(predict, data.getIdealArray());
                for(int j = 0; j < error.length; j++) {
                    validSumError += (error[j] * error[j] * data.getSignificance());
                }
            }
            TASK_LOG.info("Training error is {}, validation error is {}.", trainSumError, validSumError);

            // set cnt, error to params and return to master
            MTLParams mtlParams = new MTLParams();
            mtlParams.setTrainCount(trainCnt);
            mtlParams.setValidationCount(validCnt);
            mtlParams.setTrainSize(trainSize);
            mtlParams.setValidationSize(validationSize);
            mtlParams.setTrainError(trainSumError);
            mtlParams.setValidationError(validSumError);
            mtlParams.setMtm(this.mtm);
            return mtlParams;
        }
    }

}
