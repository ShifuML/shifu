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
package ml.shifu.shifu.core.dtrain.wnd.activation;

/**
 * Swish Activation.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class Swish extends Activation {
    /**
     * Tmp save last inputs in forward and then can be used in backward computation.
     */
    private float[] lastInput;

    @Override
    public float[] forward(float[] input) {
        this.lastInput = input;
        float[] result = new float[input.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = (float) (input[i] * (1/ (Math.exp(-1* input[i]) +1 )));
        }
        return result;
    }

    @Override
    public float[] backward(float[] backInput, float significance) {
        float[] result = new float[backInput.length];
        for(int i = 0; i < result.length; i++) {
            float sigmoid = (float) (1 / (1.0 + Math.exp(-lastInput[i])));
            result[i] = sigmoid + lastInput[i] * sigmoid * (1 - sigmoid);
        }
        return result;
    }
}
