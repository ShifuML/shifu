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
package ml.shifu.shifu.core.dtrain.wdl;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import ml.shifu.shifu.util.PermutationShuffler;
import ml.shifu.shifu.util.Shuffler;
import org.encog.mathutil.BoundMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.shifu.core.dtrain.layer.SparseInput;
import ml.shifu.shifu.core.dtrain.loss.ErrorCalculation;
import ml.shifu.shifu.core.dtrain.loss.LogErrorCalculation;
import ml.shifu.shifu.core.dtrain.loss.LossType;
import ml.shifu.shifu.core.dtrain.loss.SquaredErrorCalculation;

/**
 * {@link WDLParallelGradient} is a class design to running training process in parallel. Both batch training and
 * mini-batch training are supported by this class.
 * 
 * For batch training, call {@link WDLParallelGradient#doCompute()}
 * For mini-batch training, call {@link WDLParallelGradient#doCompute(int, int)}
 * 
 * User can configure the parameter ${@link WDLParallelGradient#threadNumber} stands for how many threads for each
 * worker in ModelConfig.json.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class WDLParallelGradient {
    private static final Logger LOG = LoggerFactory.getLogger(WDLParallelGradient.class);

    private int threadNumber;
    private WideAndDeep wdl;
    private MemoryLimitedList<WDLWorker.Data> trainData;
    private MemoryLimitedList<WDLWorker.Data> testData;
    private CompletionService<WDLParams> completionService;
    private ConcurrentMap<Integer, Integer> inputIndexMap;

    private int[] trainLows;
    private int[] trainHighs;
    private int[] testLows;
    private int[] testHighs;

    private LossType lossType;

    public WDLParallelGradient(final WideAndDeep wnd, int threadNumber, ConcurrentMap<Integer, Integer> inputIndexMap,
            final MemoryLimitedList<WDLWorker.Data> trainData, final MemoryLimitedList<WDLWorker.Data> testData,
            CompletionService<WDLParams> completionService, LossType lossType) {
        this.threadNumber = threadNumber;
        this.wdl = wnd;
        this.inputIndexMap = inputIndexMap;
        this.trainData = trainData;
        this.testData = testData;
        this.completionService = completionService;
        this.lossType = lossType;

        assert threadNumber > 0 && threadNumber < 33;
        adjustTrainSet(this.trainData.size());
        adjustTestSet(0, this.testData.size());
    }

    /**
     * In general, we adopt multi-threads to fast speed the training process. So for the whole training set, we will
     * divided it into {@link WDLParallelGradient#threadNumber} groups, each thread own one slice of training set which
     * index starts from trainLows[i] to trainHighs[i] in {@link WDLParallelGradient#trainData}.
     *
     * While for mini-batch case, instead of training all the data we just training a small part of it, namely
     * miniBatchSize, and return.
     * So to achieve this, we need to adjust the start index of training set for each iteration. And also need update
     * the slice for each thread.
     *
     * @param miniBatchSize
     *          The mini-batch size for each iteration.
     */
    private void adjustTrainSet(int miniBatchSize) {
        int recordCount = Math.min(miniBatchSize, this.trainData.size());
        if(this.trainData != null && this.trainData.size() > 0) {
            this.trainLows = new int[threadNumber];
            this.trainHighs = new int[threadNumber];

            int stepCount = Math.max(recordCount / threadNumber, 1);
            if(recordCount % threadNumber != 0) {
                stepCount += (recordCount % threadNumber) / stepCount;
            }
            for(int i = 0; i < threadNumber; i++) {
                int lowOffset = i * stepCount < recordCount ? i * stepCount : recordCount - 1;
                int highOffset = lowOffset + stepCount - 1 < recordCount ? lowOffset + stepCount - 1 : recordCount - 1;
                this.trainLows[i] = lowOffset;
                this.trainHighs[i] = highOffset;
            }
            LOG.info("Train record count: {}", recordCount);
            LOG.info("Train lows: {}", Arrays.toString(trainLows));
            LOG.info("Train highs: {}", Arrays.toString(trainHighs));
        }
    }

    private void adjustTestSet(int start, int miniBatchSize) {
        assert start >= 0;
        int recordCount = start + miniBatchSize > this.testData.size() ? this.testData.size() - start: miniBatchSize;
        if(this.testData != null && this.testData.size() > 0) {
            this.testLows = new int[threadNumber];
            this.testHighs = new int[threadNumber];

            int stepCount = Math.max(recordCount / threadNumber, 1);
            if(stepCount % threadNumber != 0) {
                // move step count to append last gap to avoid last thread worse 2*testStepCount-1
                stepCount += (recordCount % threadNumber) / stepCount;
            }
            for(int i = 0; i < threadNumber; i++) {
                int lowOffset = i * stepCount < recordCount ? i * stepCount : recordCount - 1;
                int highOffset = lowOffset + stepCount - 1 < recordCount ? lowOffset + stepCount - 1 : recordCount - 1;
                this.testLows[i] = start + lowOffset;
                this.testHighs[i] = start + highOffset;
            }

            LOG.info("Test record count: {}", recordCount);
            LOG.info("Test lows: {}", Arrays.toString(testLows));
            LOG.info("Test highs: {}", Arrays.toString(testHighs));
        }
    }

    public WDLParams doCompute() {
        return doCompute(false);
    }

    /**
     * Batch training in parallel with {@link WDLParallelGradient#threadNumber} threads
     *
     * @param shuffle whether to shuffle data before train
     * @return
     *      the combined gradients result
     */
    public WDLParams doCompute(boolean shuffle) {
        long start = System.currentTimeMillis();
        Shuffler shuffler = null;
        if (shuffle) {
            shuffler = new PermutationShuffler(this.trainData.size());
        }
        for(int i = 0; i < this.threadNumber; i++) {
            if(this.trainData != null && this.testData != null) {
                this.completionService.submit(
                        new GradientTask(this.wdl, this.inputIndexMap, this.trainData, this.testData, this.trainLows[i],
                                this.trainHighs[i], this.testLows[i], this.testHighs[i], this.lossType, shuffler));
            } else if(this.trainData != null) {
                this.completionService.submit(new GradientTask(this.wdl, this.inputIndexMap, this.trainData, null, -1,
                        -1, -1, -1, this.lossType, shuffler));
            }

        }
        WDLParams params = null;
        for(int i = 0; i < this.threadNumber; i++) {
            try {
                WDLParams paramsTmp = this.completionService.take().get();
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
     * MiniBatch training in parallel with {@link WDLParallelGradient#threadNumber} threads. For the first iteration,
     * it will training data slice [0, batchSize], and next iteration [batchSize, 2 * batchSize), and so on.
     * 
     * @param iteration
     *          the current iteration of whole training process.
     * @param miniBatchSize
     *          the mini batch size for each iteration
     * @return
     *          the combined gradients result
     */
    public WDLParams doCompute(int iteration, int miniBatchSize) {
        LOG.info("training on iteration: " + iteration + " with miniBatchSize: " + miniBatchSize);
        adjustTrainSet(miniBatchSize);
        return doCompute(true);
    }

    public static class GradientTask implements Callable<WDLParams> {
        private final Logger TASK_LOG = LoggerFactory.getLogger(GradientTask.class);
        private WideAndDeep wnd;
        private MemoryLimitedList<WDLWorker.Data> trainData;
        private MemoryLimitedList<WDLWorker.Data> testData;
        private ConcurrentMap<Integer, Integer> inputIndexMap;
        private int trainLow;
        private int trainHigh;
        private int testLow;
        private int testHigh;
        private LossType lossType;
        private final Shuffler shuffler;

        private ErrorCalculation errorCalculation;

        public GradientTask(final WideAndDeep wdl, ConcurrentMap<Integer, Integer> inputIndexMap,
                final MemoryLimitedList<WDLWorker.Data> trainData, final MemoryLimitedList<WDLWorker.Data> testData,
                int trainLow, int trainHigh, int testLow, int testHigh, LossType lossType, Shuffler shuffler) {
            this.wnd = wdl.clone();
            this.inputIndexMap = inputIndexMap;
            this.trainData = trainData;
            this.testData = testData;
            this.trainLow = trainLow;
            this.trainHigh = trainHigh;
            this.testLow = testLow;
            this.testHigh = testHigh;
            this.lossType = lossType;
            this.shuffler = shuffler;
            switch(this.lossType) {
                case LOG:
                    this.errorCalculation = new LogErrorCalculation();
                    break;
                case SQUARED:
                default:
                    this.errorCalculation = new SquaredErrorCalculation();
                    break;
            }
        }

        private List<SparseInput> getWideInputs(WDLWorker.Data data) {
            return this.wnd.getWideColumnIds().stream()
                    .map(id -> data.getCategoricalValues()[this.inputIndexMap.get(id)]).collect(Collectors.toList());
        }

        private List<SparseInput> getEmbedInputs(WDLWorker.Data data) {
            return this.wnd.getEmbedColumnIds().stream()
                    .map(id -> data.getCategoricalValues()[this.inputIndexMap.get(id)]).collect(Collectors.toList());
        }

        private double sigmoid(double logit) {
            return 1.0d / (1.0d + BoundMath.exp(-1 * logit));
        }

        @Override
        public WDLParams call() throws Exception {
            if(this.wnd == null || testHigh < testLow || trainHigh < trainLow) {
                TASK_LOG.error("input parameters not correct, testHigh={}, testLow={}, trainHigh={}, trainLow={}",
                        testHigh, testLow, trainHigh, trainLow);
                return null;
            }
            WDLParams wdlParams = new WDLParams();
            if(this.trainData.size() == 0) {
                // All field will be 0
                return wdlParams;
            }

            long start = System.currentTimeMillis();
            // forward and backward compute gradients for each iteration
            double trainCnt = trainHigh - trainLow, validCnt = testHigh - testLow;
            double trainSize = 0, validationSize = 0;
            double trainSumError = 0d, validSumError = 0d;

            int index = 0;
            for(int i = trainLow; i < trainHigh; i++) {
                WDLWorker.Data data = trainData.get(shuffler == null ? i : shuffler.getIndex(i));
                trainSize += data.getWeight();
                double[] logits = this.wnd.forward(data.getNumericalValues(), getEmbedInputs(data),
                        getWideInputs(data));
                double predict = sigmoid(logits[0]);
                trainSumError += this.errorCalculation.updateError(predict, data.getLabel()) * data.getWeight();
                this.wnd.backward(new double[] { predict }, new double[] { data.getLabel() }, data.getWeight(),
                        this.lossType);
                index += 1;
            }
            TASK_LOG.info("Worker with training time {} ms.", (System.currentTimeMillis() - start));

            start = System.currentTimeMillis();
            index = 0;
            TASK_LOG.info("Start validation computation.");
            this.wnd.setIndex(0);
            // compute validation error
            if(testData != null) {
                for(int i = testLow; i < testHigh; i++) {
                    WDLWorker.Data data = testData.get(i);
                    double[] logits = this.wnd.forward(data.getNumericalValues(), getEmbedInputs(data),
                            getWideInputs(data));
                    double sigmoid = sigmoid(logits[0]);
                    if(index++ <= 0) {
                        TASK_LOG.info("Index {}, logit {}, sigmoid {}, label {}.", index, logits[0], sigmoid,
                                data.getLabel());
                    }
                    validationSize += data.getWeight();
                    validSumError += this.errorCalculation.updateError(sigmoid, data.getLabel()) * data.getWeight();
                }
            }

            TASK_LOG.info("training error is {} {}", trainSumError, validSumError);
            // set cnt, error to params and return to master
            wdlParams.setTrainCount(trainCnt);
            wdlParams.setValidationCount(validCnt);
            wdlParams.setTrainSize(trainSize);
            wdlParams.setValidationSize(validationSize);
            wdlParams.setTrainError(trainSumError);
            wdlParams.setValidationError(validSumError);
            wdlParams.setWnd(this.wnd);

            TASK_LOG.info("Worker with validation run time {} ms.", (System.currentTimeMillis() - start));
            return wdlParams;
        }
    }
}
