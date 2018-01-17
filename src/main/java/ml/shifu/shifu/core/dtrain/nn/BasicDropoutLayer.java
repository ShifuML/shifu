/*
 * Copyright [2013-2017] PayPal Software Foundation
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

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * A new {@link BasicLayer} to add dropout support.
 */
public class BasicDropoutLayer extends BasicLayer {

    private static final long serialVersionUID = 1744506772111445808L;

    /**
     * Dropout rate in current layer.
     */
    private double dropout;

    /**
     * Construct this layer with a non-default activation function, also determine if a bias is desired or not.
     * 
     * @param activationFunction
     *            The activation function to use.
     * @param neuronCount
     *            How many neurons in this layer.
     * @param hasBias
     *            True if this layer has a bias.
     * @param dropout drop out rate
     */
    public BasicDropoutLayer(final ActivationFunction activationFunction, final boolean hasBias, final int neuronCount,
            double dropout) {
        super(activationFunction, hasBias, neuronCount);
        this.dropout = dropout;
    }

    /**
     * Construct this layer with a non-default activation function, also determine if a bias is desired or not.
     * 
     * @param activationFunction
     *            The activation function to use.
     * @param neuronCount
     *            How many neurons in this layer.
     * @param hasBias
     *            True if this layer has a bias.
     */
    public BasicDropoutLayer(final ActivationFunction activationFunction, final boolean hasBias, final int neuronCount) {
        super(activationFunction, hasBias, neuronCount);
    }

    /**
     * Construct this layer with a sigmoid activation function.
     * 
     * @param neuronCount
     *            How many neurons in this layer.
     */
    public BasicDropoutLayer(final int neuronCount) {
        this(new ActivationTANH(), true, neuronCount);
    }

    /**
     * @return the dropout
     */
    public double getDropout() {
        return dropout;
    }

    /**
     * @param dropout
     *            the dropout to set
     */
    public void setDropout(double dropout) {
        this.dropout = dropout;
    }

}
