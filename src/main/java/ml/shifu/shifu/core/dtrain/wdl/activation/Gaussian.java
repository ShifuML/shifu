/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wdl.activation;

import org.encog.mathutil.BoundMath;

/**
 * Gaussian Activation.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class Gaussian extends Activation {

    /**
     * The center
     */
    private float center;

    /**
     * The peak
     */
    private float peak;

    /**
     * The width
     */
    private float width;

    /**
     * Default construction, used for build an empty Gaussian
     */
    public Gaussian() {
    }

    public Gaussian(float center, float peak, float width) {
        this.center = center;
        this.peak = peak;
        this.width = width;
    }

    /**
     * Tmp save last inputs in forward and then can be used in backward computation.
     */
    private float[] lastInput;

    @Override
    public float[] forward(float[] inputs) {
        this.lastInput = inputs;
        float[] outputs = new float[inputs.length];
        for(int i = 0; i < inputs.length; i++) {
            outputs[i] = (float) (this.peak * BoundMath.exp(-Math.pow(inputs[i] - this.center, 2)
                    / (2.0 * this.width * this.width)));
        }
        return outputs;
    }

    @Override
    public float[] backward(float[] outputs) {
        float[] results = new float[outputs.length];
        for(int i = 0; i < outputs.length; i++) {
            float interExpValue = this.width * this.width * this.lastInput[i] * this.lastInput[i];
            results[i] = (float) (Math.exp(-0.5 * interExpValue) * peak * width * width * (interExpValue - 1));
        }
        return results;
    }

    /**
     * @return the center
     */
    public float getCenter() {
        return center;
    }

    /**
     * @param center the center to set
     */
    public void setCenter(float center) {
        this.center = center;
    }

    /**
     * @return the peak
     */
    public float getPeak() {
        return peak;
    }

    /**
     * @param peak the peak to set
     */
    public void setPeak(float peak) {
        this.peak = peak;
    }

    /**
     * @return the width
     */
    public float getWidth() {
        return width;
    }

    /**
     * @param width the width to set
     */
    public void setWidth(float width) {
        this.width = width;
    }
}
