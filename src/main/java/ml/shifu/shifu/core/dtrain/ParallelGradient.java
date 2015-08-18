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
package ml.shifu.shifu.core.dtrain;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.encog.ml.data.MLDataSet;
import org.encog.neural.error.ErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link ParallelGradient} is copied from Encog framework. The reason is that we original Gradient don't pop up
 * {@link #gradients} outside. While we need gradients accumulated into {@link NNMaster} to update NN weights.
 */
public class ParallelGradient {

    protected static final Logger LOG = LoggerFactory.getLogger(ParallelGradient.class);

    /**
     * The network to train.
     */
    private FlatNetwork network;

    /**
     * The training data.
     */
    private final MLDataSet training;

    /**
     * The testing data, test data set here is used for training and testing cross over.
     */
    private final MLDataSet testing;

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

    /**
     * The error function to use.
     */
    private final ErrorFunction errorFunction;

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
     * Construct a gradient worker.
     * 
     * @param theNetwork
     *            The network to train.
     * @param theOwner
     *            The owner that is doing the training.
     * @param theTraining
     *            The training data.
     * @param theLow
     *            The low index to use in the training data.
     * @param theHigh
     *            The high index to use in the training data.
     */
    public ParallelGradient(final FlatNetwork theNetwork, final MLDataSet theTraining, final MLDataSet theTesting,
            final double[] flatSpot, ErrorFunction ef, boolean isCrossOver, int threadCount) {
        assert threadCount > 0 && threadCount < 100;
        this.threadCount = threadCount;
        this.training = theTraining;
        long recordCount = this.training.getRecordCount();

        this.trainLows = new long[threadCount];
        this.trainHighs = new long[threadCount];
        long stepCount = recordCount / threadCount;
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
        this.errorFunction = ef;
    }

    public double[] computeGradients() {
        // TODO make theradPool as instance
        ExecutorService threadPool = Executors.newFixedThreadPool(this.threadCount);
        CompletionService<double[]> completionService = new ExecutorCompletionService<double[]>(threadPool);
        this.subGradients = new SubGradient[this.threadCount];
        try {
            for(int i = 0; i < this.threadCount; i++) {
                this.subGradients[i] = new SubGradient(this.getNetwork().clone(), this.training, this.trainLows[i],
                        this.trainHighs[i], this.testing, this.testLows[i], this.testHighs[i], this.flatSpot,
                        this.errorFunction, this.isCrossOver);
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
        } finally {
            threadPool.shutdownNow();
            try {
                threadPool.awaitTermination(2, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
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
        ExecutorService threadPool = Executors.newFixedThreadPool(this.threadCount);
        CompletionService<Double> completionService = new ExecutorCompletionService<Double>(threadPool);
        try {
            for(int i = 0; i < this.threadCount; i++) {
                final SubGradient subGradient = this.subGradients[i];
                completionService.submit(new Callable<Double>() {

                    @Override
                    public Double call() throws Exception {
                        return subGradient.caculateTotalError();
                    }
                });
            }

            int rCnt = 0;
            double errorSum = 0d;
            while(rCnt < this.threadCount) {
                try {
                    errorSum += completionService.take().get();
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                rCnt += 1;
            }

            return errorSum / (this.testing.getRecordCount() * this.getNetwork().getOutputCount());
        } finally {
            threadPool.shutdownNow();
            try {
                threadPool.awaitTermination(2, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

}
