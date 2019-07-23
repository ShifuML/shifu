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

import org.encog.mathutil.BoundMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.util.MemoryLimitedList;

/**
 * To running gradient update in parallel.
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

    public WDLParallelGradient(final WideAndDeep wnd, int threadNumber, ConcurrentMap<Integer, Integer> inputIndexMap,
            final MemoryLimitedList<WDLWorker.Data> trainData, final MemoryLimitedList<WDLWorker.Data> testData,
            CompletionService<WDLParams> completionService) {
        this.threadNumber = threadNumber;
        this.wdl = wnd;
        this.inputIndexMap = inputIndexMap;
        this.trainData = trainData;
        this.testData = testData;
        this.completionService = completionService;

        assert threadNumber > 0 && threadNumber < 33;
        int recordCount = this.trainData.size();
        this.trainLows = new int[threadNumber];
        this.trainHighs = new int[threadNumber];

        int stepCount = recordCount / threadNumber;
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

        int testRecordCount = this.testData.size();
        this.testLows = new int[threadNumber];
        this.testHighs = new int[threadNumber];
        int testStepCount = testRecordCount / threadNumber;
        if(testRecordCount % threadNumber != 0) {
            // move step count to append last gap to avoid last thread worse 2*testStepCount-1
            testStepCount += (testRecordCount % threadNumber) / testStepCount;
        }
        for(int i = 0; i < threadNumber; i++) {
            this.testLows[i] = i * testStepCount;
            if(i != threadNumber - 1) {
                this.testHighs[i] = this.testLows[i] + testStepCount - 1;
            } else {
                this.testHighs[i] = testRecordCount - 1;
            }
        }

        LOG.info("Test record count: {}", testRecordCount);
        LOG.info("Test lows: {}", Arrays.toString(testLows));
        LOG.info("Test highs: {}", Arrays.toString(testHighs));
    }

    public WDLParams doCompute() {
        long start = System.currentTimeMillis();
        for(int i = 0; i < this.threadNumber; i++) {
            this.completionService.submit(new GradientTask(this.wdl, this.inputIndexMap, this.trainData, this.testData,
                    this.trainLows[i], this.trainHighs[i], this.testLows[i], this.testHighs[i]));
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

    class GradientTask implements Callable<WDLParams> {
        private final Logger TASK_LOG = LoggerFactory.getLogger(GradientTask.class);
        private WideAndDeep wnd;
        private MemoryLimitedList<WDLWorker.Data> trainData;
        private MemoryLimitedList<WDLWorker.Data> testData;
        private ConcurrentMap<Integer, Integer> inputIndexMap;
        private int trainLow;
        private int trainHigh;
        private int testLow;
        private int testHigh;

        public GradientTask(final WideAndDeep wdl, ConcurrentMap<Integer, Integer> inputIndexMap,
                final MemoryLimitedList<WDLWorker.Data> trainData, final MemoryLimitedList<WDLWorker.Data> testData,
                int trainLow, int trainHigh, int testLow, int testHigh) {
            this.wnd = wdl.clone();
            this.inputIndexMap = inputIndexMap;
            this.trainData = trainData;
            this.testData = testData;
            this.trainLow = trainLow;
            this.trainHigh = trainHigh;
            this.testLow = testLow;
            this.testHigh = testHigh;
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
                WDLWorker.Data data = trainData.get(i);
                trainSize += data.getWeight();
                double[] logits = this.wnd.forward(data.getNumericalValues(), getEmbedInputs(data),
                        getWideInputs(data));
                double predict = sigmoid(logits[0]);
                double error = predict - data.getLabel();
                trainSumError += (error * error * data.getWeight());
                this.wnd.backward(new double[] { predict }, new double[] { data.getLabel() }, data.getWeight());
                index += 1;
            }
            TASK_LOG.info("Worker with training time {} ms.", (System.currentTimeMillis() - start));

            start = System.currentTimeMillis();
            index = 0;
            TASK_LOG.info("Start validation computation.");
            this.wnd.setIndex(0);
            // compute validation error
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
                double error = sigmoid - data.getLabel();
                validSumError += (error * error * data.getWeight());
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

            // if(this.wnd.isWideEnable()) {
            // double[] wgrads = this.wnd.getWl().getDenseLayer().getwGrads();
            // LOG.info(
            // "wgrads[159] {}, wgrads[271] {}, wgrads[320] {}, wgrads[492] {}, wgrads[516] {}, wgrads[559] {},
            // wgrads[560] {}.",
            // wgrads[159], wgrads[271], wgrads[320], wgrads[492], wgrads[516], wgrads[559], wgrads[560]);
            // } else if(this.wnd.isDeepEnable()) {
            // for(Iterator<Layer> iterator = this.wnd.getHiddenLayers().iterator(); iterator.hasNext();) {
            // Layer layer = iterator.next();
            // if(layer instanceof DenseLayer) {
            // DenseLayer dl = (DenseLayer) layer;
            // double[][] ws = dl.getWeights();
            // for(int i = 0; i < ws.length; i++) {
            // for(int j = 0; j < ws[i].length; j++) {
            // if(Math.abs(ws[i][j]) > 10) {
            // LOG.info("Hidden layer Column {}, with wegiht {} > 10, weights {}.", i, ws[i][j],
            // Arrays.toString(ws[i]));
            // }
            // }
            // }
            // break;
            // }
            // }
            //
            // // LOG.info(
            // // "wgrads[159] {}, wgrads[271] {}, wgrads[320] {}, wgrads[492] {}, wgrads[516] {}, wgrads[559] {},
            // // wgrads[560] {}.",
            // // wgrads[159][0], wgrads[271][0], wgrads[320][0], wgrads[492][0], wgrads[516][0], wgrads[559][0],
            // // wgrads[560][0]);
            // }

            TASK_LOG.info("Worker with validation run time {} ms.", (System.currentTimeMillis() - start));
            return wdlParams;
        }
    }
}
