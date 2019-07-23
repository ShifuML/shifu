package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.shifu.core.dtrain.wdl.WDLWorker;
import org.encog.mathutil.BoundMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;

/**
 * @author haillu
 * @date 7/22/2019 5:39 PM
 */
public class MTNNParallelGradient {
    private final Logger LOG = LoggerFactory.getLogger(MTNNParallelGradient.class);

    private int threadNumber;
    private MultiTaskNN mtnn;
    private MemoryLimitedList<WDLWorker.Data> trainData;
    private MemoryLimitedList<WDLWorker.Data> testData;
    private CompletionService<MTNNParams> completionService;

    private int[] trainLows;
    private int[] trainHighs;
    private int[] testLows;
    private int[] testHighs;

    public MTNNParallelGradient(int threadNumber, MultiTaskNN mtnn, MemoryLimitedList<WDLWorker.Data> trainData, MemoryLimitedList<WDLWorker.Data> testData, CompletionService<MTNNParams> completionService) {
        this.threadNumber = threadNumber;
        this.mtnn = mtnn;
        this.trainData = trainData;
        this.testData = testData;
        this.completionService = completionService;

        //just copy from WDLParallelGradient
        assert threadNumber > 0 && threadNumber < 33;
        int recordCount = this.trainData.size();
        this.trainLows = new int[threadNumber];
        this.trainHighs = new int[threadNumber];

        int steps = recordCount / threadNumber;
        if (recordCount % threadNumber != 0) {
            steps += (recordCount % threadNumber) / steps;
        }
        for (int i = 0; i < threadNumber; i++) {
            this.trainLows[i] = i * steps;
            if (i != threadNumber - 1) {
                this.trainHighs[i] = this.trainLows[i] + steps - 1;
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
        if (testRecordCount % threadNumber != 0) {
            // move step count to append last gap to avoid last thread worse 2*testStepCount-1
            testStepCount += (testRecordCount % threadNumber) / testStepCount;
        }
        for (int i = 0; i < threadNumber; i++) {
            this.testLows[i] = i * testStepCount;
            if (i != threadNumber - 1) {
                this.testHighs[i] = this.testLows[i] + testStepCount - 1;
            } else {
                this.testHighs[i] = testRecordCount - 1;
            }
        }

        LOG.info("Test record count: {}", testRecordCount);
        LOG.info("Test lows: {}", Arrays.toString(testLows));
        LOG.info("Test highs: {}", Arrays.toString(testHighs));
    }

    public MTNNParams doCompute() {
        long start = System.currentTimeMillis();
        for (int i = 0; i < this.threadNumber; i++) {
            this.completionService.submit(new MTNNParallelGradient.GradientTask(this.mtnn, this.trainData, this.testData,
                    this.trainLows[i], this.trainHighs[i], this.testLows[i], this.testHighs[i]));
        }
        MTNNParams params = null;
        for (int i = 0; i < this.threadNumber; i++) {
            try {
                MTNNParams paramsTmp = this.completionService.take().get();
                if (paramsTmp != null) {
                    if (params != null) {
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

    class GradientTask implements Callable<MTNNParams> {
        private final Logger TASK_LOG = LoggerFactory.getLogger(MTNNParallelGradient.GradientTask.class);
        private MultiTaskNN mtnn;
        private MemoryLimitedList<WDLWorker.Data> trainData;
        private MemoryLimitedList<WDLWorker.Data> testData;
        private int trainLow;
        private int trainHigh;
        private int testLow;
        private int testHigh;

        public GradientTask(MultiTaskNN mtnn, MemoryLimitedList<WDLWorker.Data> trainData, MemoryLimitedList<WDLWorker.Data> testData, int trainLow, int trainHigh, int testLow, int testHigh) {
            this.mtnn = mtnn;
            this.trainData = trainData;
            this.testData = testData;
            this.trainLow = trainLow;
            this.trainHigh = trainHigh;
            this.testLow = testLow;
            this.testHigh = testHigh;
        }

        private double sigmoid(double logit) {
            return 1.0d / (1.0d + BoundMath.exp(-1 * logit));
        }

        @Override
        public MTNNParams call() throws Exception {
            if (this.mtnn == null || testHigh < testLow || trainHigh < trainLow) {
                TASK_LOG.error("input parameters not correct, testHigh={}, testLow={}, trainHigh={}, trainLow={}",
                        testHigh, testLow, trainHigh, trainLow);
                return null;
            }
            MTNNParams mtnnParams = new MTNNParams();
            if (this.trainData.size() == 0) {
                // All field will be 0
                return mtnnParams;
            }

            long start = System.currentTimeMillis();
            // forward and backward compute gradients for each iteration
            double trainCnt = trainHigh - trainLow, validCnt = testHigh - testLow;
            double trainSize = 0, validationSize = 0;
            double trainSumError = 0d, validSumError = 0d;

            int index = 0;
            for (int i = trainLow; i < trainHigh; i++) {
                WDLWorker.Data data = trainData.get(i);
                trainSize += data.getWeight();
                double[] logits = this.mtnn.forward(data.getNumericalValues());
                double predict = sigmoid(logits[0]);
                double error = predict - data.getLabel();
                trainSumError += (error * error * data.getWeight());
                this.mtnn.backward(new double[]{predict}, new double[]{data.getLabel()}, data.getWeight());
                index += 1;
            }
            TASK_LOG.info("Worker with training time {} ms.", (System.currentTimeMillis() - start));

            start = System.currentTimeMillis();
            index = 0;
            TASK_LOG.info("Start validation computation.");

            // compute validation error
            for (int i = testLow; i < testHigh; i++) {
                WDLWorker.Data data = testData.get(i);
                double[] logits = this.mtnn.forward(data.getNumericalValues());
                double sigmoid = sigmoid(logits[0]);
                if (index++ <= 0) {
                    TASK_LOG.info("Index {}, logit {}, sigmoid {}, label {}.", index, logits[0], sigmoid,
                            data.getLabel());
                }
                validationSize += data.getWeight();
                double error = sigmoid - data.getLabel();
                validSumError += (error * error * data.getWeight());
            }

            TASK_LOG.info("training error is {} {}", trainSumError, validSumError);
            // set cnt, error to params and return to master
            mtnnParams.setTrainCount(trainCnt);
            mtnnParams.setValidationCount(validCnt);
            mtnnParams.setTrainSize(trainSize);
            mtnnParams.setValidationSize(validationSize);
            mtnnParams.setTrainError(trainSumError);
            mtnnParams.setValidationError(validSumError);
            mtnnParams.setMtnn(this.mtnn);

            TASK_LOG.info("Worker with validation run time {} ms.", (System.currentTimeMillis() - start));
            return mtnnParams;
        }
    }
}
