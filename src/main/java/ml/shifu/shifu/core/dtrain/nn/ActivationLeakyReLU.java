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
 * Leaky Rectified linear unit. Default alpha=0.01, cutoff=0
 * Out(x) = alpha * x if x
 * Out(x) = x if x bigger or equal to 0
 * Leaky ReLU may avoid zero gradient "dying ReLU" problem by having non-zero gradient below 0.
 * 
 * See for example http://arxiv.org/abs/1505.00853 for a comparison of ReLU variants.
 */
public class ActivationLeakyReLU implements ActivationFunction {

    /**
     * The serial ID.
     */
    private static final long serialVersionUID = -6010599783506925583L;

    /**
     * The ramp low threshold parameter.
     */
    public static final int PARAM_RELU_LOW_THRESHOLD = 0;

    /**
     * The alpha parameter. Default alpha is 0.01
     */
    public static final int PARAM_LEAKY_RELU_ALPHA = 1;
    
    /**
     * The parameters.
     */
    private final double[] params;

    /**
     * Default constructor.
     */
    public ActivationLeakyReLU() {
        this(0, 0.01);
    }

    /**
     * Construct a ramp activation function.
     * 
     * @param thresholdLow
     *            The low threshold value.
     * @param alpha
     *            The alpha value will be timed if the low threshold is exceeded.
     */
    public ActivationLeakyReLU(final double thresholdLow, final double alpha) {
        this.params = new double[2];
        this.params[ActivationLeakyReLU.PARAM_RELU_LOW_THRESHOLD] = thresholdLow;
        this.params[ActivationLeakyReLU.PARAM_LEAKY_RELU_ALPHA] = alpha;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void activationFunction(final double[] x, final int start, final int size) {
        for(int i = start; i < start + size; i++) {
            if(x[i] <= this.params[ActivationLeakyReLU.PARAM_RELU_LOW_THRESHOLD]) {
                x[i] = x[i] * this.params[ActivationLeakyReLU.PARAM_LEAKY_RELU_ALPHA];
            }
        }
    }

    /**
     * Clone the object.
     * 
     * @return The cloned object.
     */
    @Override
    public final ActivationFunction clone() {
        return new ActivationLeakyReLU(this.params[ActivationLeakyReLU.PARAM_RELU_LOW_THRESHOLD],
                this.params[ActivationLeakyReLU.PARAM_LEAKY_RELU_ALPHA]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final double derivativeFunction(final double b, final double a) {
        if(b <= this.params[ActivationLeakyReLU.PARAM_RELU_LOW_THRESHOLD]) {
            return this.params[ActivationLeakyReLU.PARAM_LEAKY_RELU_ALPHA];
        }
        return 1.0;
    }

    /**
     * @return the Alpha
     */
    public final double getAlpha() {
        return this.params[ActivationLeakyReLU.PARAM_LEAKY_RELU_ALPHA];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final String[] getParamNames() {
        final String[] result = { "thresholdLow", "alpha" };
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
     * @return the thresholdLow
     */
    public final double getThresholdLow() {
        return this.params[ActivationLeakyReLU.PARAM_RELU_LOW_THRESHOLD];
    }

    /**
     * @return True, as this function does have a derivative.
     */
    @Override
    public final boolean hasDerivative() {
        return true;
    }

    /**
     * Set the low value.
     * 
     * @param d
     *            The low value.
     */
    public final void setAlpha(final double d) {
        setParam(ActivationLeakyReLU.PARAM_LEAKY_RELU_ALPHA, d);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setParam(final int index, final double value) {
        this.params[index] = value;
    }

    /**
     * Set the threshold low.
     * 
     * @param d
     *            The threshold low.
     */
    public final void setThresholdLow(final double d) {
        setParam(ActivationLeakyReLU.PARAM_RELU_LOW_THRESHOLD, d);
    }

}