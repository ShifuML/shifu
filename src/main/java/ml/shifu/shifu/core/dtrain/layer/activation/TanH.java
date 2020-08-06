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
package ml.shifu.shifu.core.dtrain.layer.activation;

/**
 * TanH Activation.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class TanH extends Activation {
    @Override
    public double[] forward(double[] input) {
        double[] outputs = new double[input.length];
        for(int i = 0; i < input.length; i++) {
            outputs[i] = (double) Math.tanh(input[i]);
        }
        return outputs;
    }

    @Override
    public double[] backward(double[] backInput) {
        double[] results = new double[backInput.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = 1 - backInput[i] * backInput[i];
        }
        return results;
    }
}
