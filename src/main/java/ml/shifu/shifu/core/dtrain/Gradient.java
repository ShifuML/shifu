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

import ml.shifu.shifu.core.dtrain.nn.NNMaster;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.error.ErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;

/**
 * {@link Gradient} is copied from Encog framework. The reason is that we original Gradient don't pop up
 * gradients outside. While we need gradients accumulated into {@link NNMaster} to update NN weights.
 */
public class Gradient {

    /**
     * The network to train.
     */
    private FlatNetwork network;

    /**
     * The error calculation method.
     */
    private final ErrorCalculation errorCalculation = new ErrorCalculation();

    /**
     * The actual values from the neural network.
     */
    private final double[] actual;

    /**
     * The deltas for each layer.
     */
    private final double[] layerDelta;

    /**
     * The neuron counts, per layer.
     */
    private final int[] layerCounts;

    /**
     * The feed counts, per layer.
     */
    private final int[] layerFeedCounts;

    /**
     * The layer indexes.
     */
    private final int[] layerIndex;

    /**
     * The index to each layer's weights and thresholds.
     */
    private final int[] weightIndex;

    /**
     * The output from each layer.
     */
    private final double[] layerOutput;

    /**
     * The sums.
     */
    private final double[] layerSums;

    /**
     * The gradients.
     */
    private double[] gradients;

    /**
     * The weights and thresholds.
     */
    private double[] weights;

    /**
     * The pair to use for training.
     */
    private final MLDataPair pair;

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
     * error
     */
    private double error;

    /**
     * Derivative add constant. Used to combat flat spot.
     */
    private double[] flatSpot;

    /**
     * The error function to use.
     */
    private final ErrorFunction errorFunction;

    public Gradient(final FlatNetwork theNetwork, final MLDataSet theTraining, final MLDataSet theTesting,
            final double[] flatSpot, ErrorFunction ef, boolean isCrossOver) {
        this.network = theNetwork;
        this.training = theTraining;
        this.testing = theTesting;
        this.isCrossOver = isCrossOver;
        this.flatSpot = flatSpot;
        this.errorFunction = ef;

        this.layerDelta = new double[getNetwork().getLayerOutput().length];
        this.gradients = new double[getNetwork().getWeights().length];
        this.actual = new double[getNetwork().getOutputCount()];

        this.weights = getNetwork().getWeights();
        this.layerIndex = getNetwork().getLayerIndex();
        this.layerCounts = getNetwork().getLayerCounts();
        this.weightIndex = getNetwork().getWeightIndex();
        this.layerOutput = getNetwork().getLayerOutput();
        this.layerSums = getNetwork().getLayerSums();
        this.layerFeedCounts = getNetwork().getLayerFeedCounts();

        this.pair = BasicMLDataPair.createPair(getNetwork().getInputCount(), getNetwork().getOutputCount());
    }

    /**
     * Process one training set element.
     * 
     * @param input
     *            The network input.
     * @param ideal
     *            The ideal values.
     * @param s
     *            The significance.
     */
    private void process(final double[] input, final double[] ideal, double s) {
        this.getNetwork().compute(input, this.actual);

        this.errorCalculation.updateError(this.actual, ideal, s);

        this.errorFunction.calculateError(ideal, actual, this.getLayerDelta());

        for(int i = 0; i < this.actual.length; i++) {
            this.getLayerDelta()[i] = ((this.getNetwork().getActivationFunctions()[0].derivativeFunction(
                    this.layerSums[i], this.layerOutput[i]) + this.flatSpot[0])) * (this.getLayerDelta()[i] * s);
        }

        for(int i = this.getNetwork().getBeginTraining(); i < this.getNetwork().getEndTraining(); i++) {
            processLevel(i);
        }
    }

    /**
     * Process one level.
     * 
     * @param currentLevel
     *            The level.
     */
    private void processLevel(final int currentLevel) {
        final int fromLayerIndex = this.layerIndex[currentLevel + 1];
        final int toLayerIndex = this.layerIndex[currentLevel];
        final int fromLayerSize = this.layerCounts[currentLevel + 1];
        final int toLayerSize = this.layerFeedCounts[currentLevel];

        final int index = this.weightIndex[currentLevel];
        final ActivationFunction activation = this.getNetwork().getActivationFunctions()[currentLevel + 1];
        final double currentFlatSpot = this.flatSpot[currentLevel + 1];

        // handle weights
        int yi = fromLayerIndex;
        for(int y = 0; y < fromLayerSize; y++) {
            final double output = this.layerOutput[yi];
            double sum = 0;
            int xi = toLayerIndex;
            int wi = index + y;
            for(int x = 0; x < toLayerSize; x++) {
                this.gradients[wi] += output * this.getLayerDelta()[xi];
                sum += this.weights[wi] * this.getLayerDelta()[xi];
                wi += fromLayerSize;
                xi++;
            }

            this.getLayerDelta()[yi] = sum
                    * (activation.derivativeFunction(this.layerSums[yi], this.layerOutput[yi]) + currentFlatSpot);
            yi++;
        }
    }

    /**
     * Perform the gradient calculation
     */
    public final void run() {
        try {
            // reset errors and gradients firstly
            this.errorCalculation.reset();
            Arrays.fill(this.gradients, 0.0);

            for(int i = 0; i < this.training.getRecordCount(); i++) {
                if(this.isCrossOver) {
                    // 3:1 to select testing data set, tmp hard code, TODO fix hard code issue,extract such logic to a
                    // method
                    if((i + seed) % 4 < 3) {
                        this.training.getRecord(i, this.pair);
                    } else {
                        long testingSize = this.testing.getRecordCount();
                        if(i < testingSize) {
                            this.testing.getRecord(i, this.pair);
                        } else {
                            this.testing.getRecord(i % testingSize, this.pair);
                        }
                    }
                } else {
                    this.training.getRecord(i, this.pair);
                }
                process(this.pair.getInputArray(), this.pair.getIdealArray(), pair.getSignificance());
            }
            this.error = this.errorCalculation.calculate();
        } catch (final Throwable ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * Calculate the error for this neural network. The error is calculated
     * using root-mean-square(RMS).
     * 
     * @return The error percentage.
     */
    public final double calculateError() {
        final ErrorCalculation errorCalculation = new ErrorCalculation();

        final double[] actual = new double[this.getNetwork().getOutputCount()];
        final MLDataPair pair = BasicMLDataPair.createPair(testing.getInputSize(), testing.getIdealSize());

        for(int i = 0; i < testing.getRecordCount(); i++) {
            if(this.isCrossOver) {
                // 3:1 to select testing data set, tmp hard code, TODO fix hard code issue
                if((i + seed) % 4 < 3) {
                    this.testing.getRecord(i, pair);
                } else {
                    long trainingSize = this.training.getRecordCount();
                    if(i < trainingSize) {
                        this.training.getRecord(i, pair);
                    } else {
                        this.training.getRecord(i % trainingSize, pair);
                    }
                }
            } else {
                this.testing.getRecord(i, pair);
            }
            this.getNetwork().compute(pair.getInputArray(), actual);
            errorCalculation.updateError(actual, pair.getIdealArray(), pair.getSignificance());
        }
        return errorCalculation.calculate();
    }

    public ErrorCalculation getErrorCalculation() {
        return errorCalculation;
    }

    /**
     * @return the gradients
     */
    public double[] getGradients() {
        return this.gradients;
    }

    /**
     * @return the error
     */
    public double getError() {
        return error;
    }

    /**
     * @return the weights
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(double[] weights) {
        this.weights = weights;
        this.getNetwork().setWeights(weights);
    }

    public void setParams(BasicNetwork network) {
        this.network = network.getFlat();
        this.weights = network.getFlat().getWeights();
    }

    public FlatNetwork getNetwork() {
        return network;
    }

    public double[] getLayerDelta() {
        return layerDelta;
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

}
