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

/**
 * LeakyReLU Activation
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class LeakyReLU extends Activation {

    /**
     * The thresholdLow
     */
    private float thresholdLow;

    /**
     * The alpha
     */
    private float alpha;

    /**
     * Tmp save last inputs in forward and then can be used in backward computation.
     */
    private float[] lastInput;

    public LeakyReLU() {
    }

    public LeakyReLU(final float thresholdLow, final float alpha) {
        this.thresholdLow = thresholdLow;
        this.alpha = alpha;
    }

    @Override
    public float[] forward(float[] inputs) {
        this.lastInput = inputs;
        float[] outputs = new float[inputs.length];
        for(int i = 0; i < inputs.length; i++) {
            if(inputs[i] <= this.thresholdLow) {
                outputs[i] = inputs[i] * this.alpha;
            }
        }
        return outputs;
    }

    @Override
    public float[] backward(float[] backInput, float significance) {
        float[] results = new float[backInput.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = this.lastInput[i] <= this.thresholdLow ? this.alpha : 1.0f;
        }
        return results;
    }

    /**
     * @return the thresholdLow
     */
    public float getThresholdLow() {
        return thresholdLow;
    }

    /**
     * @param thresholdLow
     *          the thresholdLow to set
     */
    public void setThresholdLow(float thresholdLow) {
        this.thresholdLow = thresholdLow;
    }

    /**
     * @return the alpha
     */
    public float getAlpha() {
        return alpha;
    }

    /**
     * @param alpha
     *          #the alpha to set
     */

    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }
}
