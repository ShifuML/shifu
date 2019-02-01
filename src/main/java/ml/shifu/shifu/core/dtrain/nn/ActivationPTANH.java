package ml.shifu.shifu.core.dtrain.nn;

import org.encog.engine.network.activation.ActivationFunction;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

/**
 * Penalized tanh activation function
 * f(x) = tanh(x)  if x &gt; 0
 *        0.25 * tanh(x) if(x &lt;=0)
 *
 * Some study already proved that Penalized tanh could generate more stable performance.
 * See https://arxiv.org/pdf/1901.02671.pdf for detail
 */

public class ActivationPTANH implements ActivationFunction {

    /**
     * Serial id for this class.
     */
    private static final long serialVersionUID = 9121998892720207643L;

    /**
     * The parameters.
     */
    private final double[] params;

    /**
     * Construct a basic HTAN activation function, with a slope of 1.
     */
    public ActivationPTANH() {
        this.params = new double[0];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void activationFunction(final double[] x, final int start,
            final int size) {
        for (int i = start; i < start + size; i++) {
            if ( x[i] > 0 ) {
                x[i] = Math.tanh(x[i]);
            } else {
                x[i] = 0.25d * Math.tanh(x[i]);
            }
        }
    }

    /**
     * @return The object cloned;
     */
    @Override
    public final ActivationFunction clone() {
        return new ml.shifu.shifu.core.dtrain.nn.ActivationPTANH();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final double derivativeFunction(final double b, final double a) {
        if ( b > 0 ) {
            return (1.0 - a * a);
        } else {
            // a = 0.25 * tanh(x)
            return 0.25d * (1.0 - 16.0d * a * a);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final String[] getParamNames() {
        final String[] result = {};
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
     * @return Return true, TANH has a derivative.
     */
    @Override
    public final boolean hasDerivative() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setParam(final int index, final double value) {
        this.params[index] = value;
    }

}
