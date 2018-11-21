/*
 * Encog(tm) Core v3.4 - Java Version
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-core

 * Copyright 2008-2016 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For more information on Heaton Research copyrights, licenses
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package ml.shifu.shifu.core.dtrain.nn;

import org.encog.engine.network.activation.ActivationFunction;

/**
 * Swish activation function is proposed by google researcher in paper - https://arxiv.org/pdf/1710.05941.pdf
 * It is defined as f(x) = x* sigmoid(x)
 * The comparison to Relu is available at https://medium.com/@jaiyamsharma/swish-in-depth-a-comparison-of-swish-relu-on-cifar-10-1c798e70ee08
 */
public class ActivationSwish implements ActivationFunction {

    /**
     * The serial ID.
     */
    private static final long serialVersionUID = 6336245112244386239L;

    /**
     * The parameters.
     */
    private final double[] params;

    public ActivationSwish() { this.params = new double[0]; }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void activationFunction(final double[] x, final int start, final int size) {
        for(int i = start; i < start + size; i++) {
            x[i] = x[i] * (1/ (Math.exp(-1* x[i]) +1 )) ;
        }
    }

    /**
     * Clone the object.
     *
     * @return The cloned object.
     */
    @Override
    public final ActivationFunction clone() {
        return new ActivationSwish();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final double derivativeFunction(final double b, final double a) {
        double sigmoid = 1 / (1.0 + Math.exp(-b));
        return sigmoid + b * sigmoid * (1 - sigmoid);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final String[] getParamNames() {
        final String[] result = { };
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final double[] getParams() {
        return this.params;
    }

    /**
     * @return True, as this function does have a derivative.
     */
    @Override
    public final boolean hasDerivative() { return true; }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setParam(final int index, final double value) {
        this.params[index] = value;
    }

}