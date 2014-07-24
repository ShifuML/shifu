/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.dtrain;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.error.ErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;

import java.util.Arrays;

/**
 * {@link Gradient} is copied from Encog framework. The reason is that we original Gradient don't pop up
 * {@link #gradients} outside. While we need gradients accumulated into {@link NNMaster} to update NN weights.
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

    /**
     * Construct a gradient worker.
     *
     * @param theNetwork  The network to train.
     * @param theOwner    The owner that is doing the training.
     * @param theTraining The training data.
     * @param theLow      The low index to use in the training data.
     * @param theHigh     The high index to use in the training data.
     */
    public Gradient(final FlatNetwork theNetwork, final MLDataSet theTraining, final double[] flatSpot, ErrorFunction ef) {
        this.network = theNetwork;
        this.training = theTraining;
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
     * @param input The network input.
     * @param ideal The ideal values.
     * @param s     The significance.
     */
    private void process(final double[] input, final double[] ideal, double s) {
        this.getNetwork().compute(input, this.actual);

        this.errorCalculation.updateError(this.actual, ideal, s);
        this.errorFunction.calculateError(ideal, actual, this.getLayerDelta());

        for (int i = 0; i < this.actual.length; i++) {
            this.getLayerDelta()[i] = ((this.getNetwork().getActivationFunctions()[0].derivativeFunction(
                    this.layerSums[i], this.layerOutput[i]) + this.flatSpot[0])) * (this.getLayerDelta()[i] * s);
        }

        for (int i = this.getNetwork().getBeginTraining(); i < this.getNetwork().getEndTraining(); i++) {
            processLevel(i);
        }
    }

    /**
     * Process one level.
     *
     * @param currentLevel The level.
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
        for (int y = 0; y < fromLayerSize; y++) {
            final double output = this.layerOutput[yi];
            double sum = 0;
            int xi = toLayerIndex;
            int wi = index + y;
            for (int x = 0; x < toLayerSize; x++) {
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

            for (int i = 0; i < this.training.getRecordCount(); i++) {
                this.training.getRecord(i, this.pair);
                process(this.pair.getInputArray(), this.pair.getIdealArray(), pair.getSignificance());
            }
            this.error = this.errorCalculation.calculate();

        } catch (final Throwable ex) {
            throw new RuntimeException(ex);
        }
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
     * @param weights the weights to set
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

}
