package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import org.encog.mathutil.BoundMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;

/**
 * @author haillu
 */
public class MTNNParallelGradient {
    private final Logger LOG = LoggerFactory.getLogger(MTNNParallelGradient.class);

    private int threadNumber;
    private MultiTaskNN mtnn;
    private MemoryLimitedList<MTNNWorker.Data> trainData;
    private MemoryLimitedList<MTNNWorker.Data> testData;
    private CompletionService<MTNNParams> completionService;

    private int[] trainLows;
    private int[] trainHighs;
    private int[] testLows;
    private int[] testHighs;

    public MTNNParallelGradient(int threadNumber, MultiTaskNN mtnn, MemoryLimitedList<MTNNWorker.Data> trainData, MemoryLimitedList<MTNNWorker.Data> testData, CompletionService<MTNNParams> completionService) {
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

        int stepLength = Math.max(recordCount / threadNumber, 1);
        // we must consider the situation that this iteration has no training data.
        if (this.trainData != null && this.trainData.size() > 0) {
            if (recordCount % threadNumber != 0) {
                stepLength += (recordCount % threadNumber) / stepLength;
            }
            for (int i = 0; i < threadNumber; i++) {
                this.trainLows[i] = i * stepLength < recordCount ? i * stepLength : recordCount - 1;
                this.trainHighs[i] = this.trainLows[i] + stepLength - 1 < recordCount ?
                        this.trainLows[i] + stepLength - 1 : recordCount - 1;
            }
            LOG.info("Train record count: {}", recordCount);
            LOG.info("Train lows: {}", Arrays.toString(trainLows));
            LOG.info("Train highs: {}", Arrays.toString(trainHighs));
        }

        if (this.testData != null && this.testData.size() > 0) {
            int testRecordCount = this.testData.size();
            this.testLows = new int[threadNumber];
            this.testHighs = new int[threadNumber];
            int testStepCount = Math.max(testRecordCount / threadNumber, 1);
            if (testRecordCount % threadNumber != 0) {
                // move step count to append last gap to avoid last thread worse 2*testStepCount-1
                testStepCount += (testRecordCount % threadNumber) / testStepCount;
            }
            for (int i = 0; i < threadNumber; i++) {
                this.testLows[i] = i * testStepCount < testRecordCount ? i * testStepCount : testRecordCount - 1;
                this.testHighs[i] = this.testLows[i] + testStepCount - 1 < testRecordCount ?
                        this.testLows[i] + testStepCount - 1 : testRecordCount - 1;
            }

            LOG.info("Test record count: {}", testRecordCount);
            LOG.info("Test lows: {}", Arrays.toString(testLows));
            LOG.info("Test highs: {}", Arrays.toString(testHighs));
        }
    }

    public MTNNParams doCompute() {
        long start = System.currentTimeMillis();
        for (int i = 0; i < this.threadNumber; i++) {
            if (trainData != null && testData != null) {
                this.completionService.submit(new GradientTask(this.mtnn, this.trainData, this.testData,
                        this.trainLows[i], this.trainHighs[i], this.testLows[i], this.testHighs[i]));
            } else if (trainData != null) {
                this.completionService.submit(new GradientTask(this.mtnn, this.trainData, null,
                        -1, -1, -1, -1));
            }

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
        private final Logger TASK_LOG = LoggerFactory.getLogger(GradientTask.class);
        private MultiTaskNN mtnn;
        private MemoryLimitedList<MTNNWorker.Data> trainData;
        private MemoryLimitedList<MTNNWorker.Data> testData;
        private int trainLow;
        private int trainHigh;
        private int testLow;
        private int testHigh;

        public GradientTask(MultiTaskNN mtnn, MemoryLimitedList<MTNNWorker.Data> trainData, MemoryLimitedList<MTNNWorker.Data> testData, int trainLow, int trainHigh, int testLow, int testHigh) {
            this.mtnn = mtnn.clone();
            this.trainData = trainData;
            this.testData = testData;
            this.trainLow = trainLow;
            this.trainHigh = trainHigh;
            this.testLow = testLow;
            this.testHigh = testHigh;
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
            double[] trainSumError = {};
            double[] validSumError = {};

            int index = 0;
            for (int i = trainLow; i < trainHigh; i++) {
                MTNNWorker.Data data = trainData.get(i);
                trainSize += data.getWeight();
                double[] predicts = this.mtnn.forward(data.getInputs());
                double[] actuals = data.getLabels();

                AssertUtils.assertEquals(predicts.length, actuals.length);
                if (i == trainLow) {
                    trainSumError = new double[predicts.length];
                }
                double[] errors = new double[predicts.length];
                for (int j = 0; j < predicts.length; j++) {
                    errors[j] = predicts[j] - actuals[j];
                    trainSumError[j] += (errors[j] * errors[j] * data.getWeight());
                }
                this.mtnn.backward(predicts, data.getLabels(), data.getWeight());
                index += 1;
            }
            TASK_LOG.info("Worker with training time {} ms.", (System.currentTimeMillis() - start));

            start = System.currentTimeMillis();
            index = 0;
            TASK_LOG.info("Start validation computation.");

            // compute validation error
            if (testData != null) {
                for (int i = testLow; i < testHigh; i++) {
                    MTNNWorker.Data data = testData.get(i);
                    validationSize += data.getWeight();
                    double[] predicts = this.mtnn.forward(data.getInputs());
                    double[] actuals = data.getLabels();

                    AssertUtils.assertEquals(predicts.length, actuals.length);
//                if (index++ <= 0) {
//                    TASK_LOG.info("Index {}, logit {}, sigmoid {}, label {}.", index, logits[0], sigmoid,
//                            data.getLabels());
//                }
                    if (i == testLow) {
                        validSumError = new double[predicts.length];
                    }
                    double[] errors = new double[predicts.length];
                    for (int j = 0; j < predicts.length; j++) {
                        errors[j] = predicts[j] - actuals[j];
                        validSumError[j] += (errors[j] * errors[j] * data.getWeight());
                    }

                }
            }

            TASK_LOG.info("trainSumError is {}, validSumError is {}", trainSumError, validSumError);
            // set cnt, error to params and return to master
            mtnnParams.setTrainCount(trainCnt);
            mtnnParams.setValidationCount(validCnt);
            mtnnParams.setTrainSize(trainSize);
            mtnnParams.setValidationSize(validationSize);
            mtnnParams.setTrainErrors(trainSumError);
            mtnnParams.setValidationErrors(validSumError);
            mtnnParams.setMtnn(this.mtnn);

            TASK_LOG.info("Worker with validation run time {} ms.", (System.currentTimeMillis() - start));
            return mtnnParams;
        }
    }
}
