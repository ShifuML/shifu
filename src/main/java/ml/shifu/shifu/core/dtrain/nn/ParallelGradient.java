/**
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.nn;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ml.shifu.shifu.core.dtrain.dataset.FloatFlatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataSet;
import ml.shifu.shifu.util.ClassUtils;

import org.encog.neural.error.ErrorFunction;
import org.encog.neural.error.LinearErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link ParallelGradient} is copied from Encog framework. The reason is that we original Gradient don't pop up
 * gradients outside. While we need gradients accumulated into {@link NNMaster} to update NN weights.
 */
public class ParallelGradient {

    protected static final Logger LOG = LoggerFactory.getLogger(ParallelGradient.class);

    /**
     * The network to train.
     */
    private FloatFlatNetwork network;

    /**
     * The training data.
     */
    private final FloatMLDataSet training;

    /**
     * The testing data, test data set here is used for training and testing cross over.
     */
    private final FloatMLDataSet testing;

    /**
     * Whether to replace training and testing elements.
     */
    private final boolean isCrossOver;

    /**
     * Seed used to sample training and testing data set to choose which element is used for training
     */
    private long seed = System.currentTimeMillis();

    /**
     * Derivative add constant. Used to combat flat spot.
     */
    private double[] flatSpot;

    private final int threadCount;

    private long[] trainLows;

    private long[] trainHighs;

    private double trainError;

    @SuppressWarnings("unused")
    private double testError;

    private long[] testLows;

    private long[] testHighs;

    private SubGradient[] subGradients;

    /**
     * Create a thread pool to do gradient computing and test set error computing using multiple threads.
     */
    private ExecutorService threadPool;

    /**
     * If enabled by extreme learning machine: https://en.wikipedia.org/wiki/Extreme_learning_machine
     */
    private boolean isELM;

    /**
     * Loss string definition: log, squared, absolute
     */
    private String lossStr;

    /**
     * If miniBatchRate set to 0.1d, {@link #batchs} is 10. It will run 10x iterations for one epochs.
     */
    private int batchs = 1;

    public ParallelGradient(final FloatFlatNetwork theNetwork, final FloatMLDataSet theTraining,
            final FloatMLDataSet theTesting, final double[] flatSpot, ErrorFunction ef, boolean isCrossOver,
            int threadCount, boolean isELM, String lossStr, int batchs) {
        this.isELM = isELM;
        assert threadCount > 0 && threadCount < 33;
        this.threadCount = threadCount;
        this.training = theTraining;
        long recordCount = this.training.getRecordCount();

        this.trainLows = new long[threadCount];
        this.trainHighs = new long[threadCount];

        // TODO not very good for such case: 80% in memory, 20% in disk, while all in disk are split into one thread
        long stepCount = recordCount / threadCount;
        if(recordCount % threadCount != 0) {
            // move step count to append last gap to avoid last thread worse 2*stepCount-1
            stepCount += (recordCount % threadCount) / stepCount;
        }
        for(int i = 0; i < threadCount; i++) {
            this.trainLows[i] = i * stepCount;
            if(i != threadCount - 1) {
                this.trainHighs[i] = this.trainLows[i] + stepCount - 1;
            } else {
                this.trainHighs[i] = recordCount - 1;
            }
        }
        LOG.info("Train record count: {}", recordCount);
        LOG.info("Train lows: {}", Arrays.toString(trainLows));
        LOG.info("Train highs: {}", Arrays.toString(trainHighs));

        this.testing = theTesting;
        long testRecordCount = this.testing.getRecordCount();

        this.testLows = new long[threadCount];
        this.testHighs = new long[threadCount];
        long testStepCount = testRecordCount / threadCount;
        if(testRecordCount % threadCount != 0) {
            // move step count to append last gap to avoid last thread worse 2*testStepCount-1
            testStepCount += (testRecordCount % threadCount) / testStepCount;
        }
        for(int i = 0; i < threadCount; i++) {
            this.testLows[i] = i * testStepCount;
            if(i != threadCount - 1) {
                this.testHighs[i] = this.testLows[i] + testStepCount - 1;
            } else {
                this.testHighs[i] = testRecordCount - 1;
            }
        }

        LOG.info("Test record count: {}", testRecordCount);
        LOG.info("Test lows: {}", Arrays.toString(testLows));
        LOG.info("Test highs: {}", Arrays.toString(testHighs));

        this.network = theNetwork;
        this.isCrossOver = isCrossOver;
        this.flatSpot = flatSpot;
        this.threadPool = Executors.newFixedThreadPool(this.threadCount);
        this.lossStr = lossStr;
        this.batchs = batchs;
    }

    public double[] computeGradients(int currentIteration) {
        CompletionService<double[]> completionService = new ExecutorCompletionService<double[]>(this.threadPool);
        this.subGradients = new SubGradient[this.threadCount];
        Random dropoutRandom = new Random();
        for(int i = 0; i < this.threadCount; i++) {
            if(this.subGradients[i] == null) {
                this.subGradients[i] = new SubGradient(this.network.clone(), this.training, this.trainLows[i],
                        this.trainHighs[i], this.testing, this.testLows[i], this.testHighs[i], this.flatSpot,
                        this.isCrossOver, this, dropoutRandom, this.batchs, currentIteration);
            } else {
                this.subGradients[i].setNetwork(this.network.clone());
            }
            this.subGradients[i].setSeed(this.getSeed());
            completionService.submit(this.subGradients[i]);
        }

        int rCnt = 0;
        double[] finalGradients = new double[this.getNetwork().getWeights().length];
        while(rCnt < this.threadCount) {
            double[] gradients = null;
            try {
                gradients = completionService.take().get();
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            for(int i = 0; i < finalGradients.length; i++) {
                finalGradients[i] += gradients[i];
            }
            rCnt += 1;
        }

        double errorSum = 0d;
        for(int i = 0; i < this.threadCount; i++) {
            errorSum += this.subGradients[i].getError() * (trainHighs[i] - trainLows[i] + 1)
                    * this.getNetwork().getOutputCount();
        }
        this.trainError = errorSum / (this.training.getRecordCount() * this.getNetwork().getOutputCount());
        return finalGradients;
    }

    /**
     * @return the seed
     */
    public long getSeed() {
        return seed;
    }

    /**
     * @param seed
     *            the seed to set
     */
    public void setSeed(long seed) {
        this.seed = seed;
    }

    /**
     * @return the trainError
     */
    public double getTrainError() {
        return trainError;
    }

    /**
     * @return the network
     */
    public FlatNetwork getNetwork() {
        return network;
    }

    public double calculateError() {
        CompletionService<Double> completionService = new ExecutorCompletionService<Double>(this.threadPool);
        final ml.shifu.shifu.core.dtrain.nn.ErrorCalculation ec = createECInstance();
        for(int i = 0; i < this.threadCount; i++) {
            final SubGradient subGradient = this.subGradients[i];
            completionService.submit(new Callable<Double>() {

                @Override
                public Double call() throws Exception {
                    return subGradient.calculateError(ec);
                }
            });
        }

        int rCnt = 0;
        while(rCnt < this.threadCount) {
            try {
                completionService.take().get();
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            rCnt += 1;
        }

        return ec.calculate();
    }

    /**
     * Create error calculation instance according to lossStr;
     * 
     * @return the ErrorCalculation instance.
     */
    public ml.shifu.shifu.core.dtrain.nn.ErrorCalculation createECInstance() {
        ml.shifu.shifu.core.dtrain.nn.ErrorCalculation ec = new SquaredErrorCalculation();
        if(lossStr.equalsIgnoreCase("log")) {
            ec = new LogErrorCalculation();
        } else if(lossStr.equalsIgnoreCase("absolute")) {
            ec = new AbsoluteErrorCalculation();
        } else if(lossStr.equalsIgnoreCase("squared")) {
            ec = new SquaredErrorCalculation();
        } else {
            try {
                ec = (ml.shifu.shifu.core.dtrain.nn.ErrorCalculation) ClassUtils.newInstance(Class.forName(lossStr));
            } catch (ClassNotFoundException e) {
                LOG.warn("Class not found for {}, using default SquaredLoss", lossStr);
                ec = new SquaredErrorCalculation();
            }
        }
        return ec;
    }

    /**
     * Create error function instance according to lossStr;
     * 
     * @return the ErrorCalculation instance.
     */
    public ErrorFunction createEFInstance() {
        ErrorFunction ef = new LinearErrorFunction();
        if(lossStr.equalsIgnoreCase("log")) {
            ef = new LogErrorFunction();
        } else if(lossStr.equalsIgnoreCase("absolute")) {
            ef = new AbsoluteErrorFunction();
        } else if(lossStr.equalsIgnoreCase("squared")) {
            ef = new LinearErrorFunction();
        } else {
            ef = new LinearErrorFunction();
        }
        return ef;
    }

    /**
     * Average weights for all sub gradients and then set to current network.
     */
    public void resetNetworkWeights() {
        double[] weights = new double[this.network.getWeights().length];
        for(int i = 0; i < subGradients.length; i++) {
            double[] subWeights = subGradients[i].getNetwork().getWeights();
            for(int j = 0; j < weights.length; j++) {
                weights[j] += subWeights[j];
            }
        }
        for(int j = 0; j < weights.length; j++) {
            weights[j] /= subGradients.length;
        }
        this.network.setWeights(weights);
    }

    /**
     * Shut down thread pool, should be called at last to make sure jvm exit
     */
    public void shutdown() {
        this.threadPool.shutdownNow();
        try {
            this.threadPool.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * @return the isELM
     */
    public boolean isELM() {
        return isELM;
    }

}
