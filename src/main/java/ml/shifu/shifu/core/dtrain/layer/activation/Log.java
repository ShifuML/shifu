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

import org.encog.mathutil.BoundMath;

/**
 * Log Activation
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class Log extends Activation {
    /**
     * Tmp save last inputs in forward and then can be used in backward computation.
     */
    private double[] lastInput;

    @Override
    public double[] forward(double[] inputs) {
        this.lastInput = inputs;
        double[] outputs = new double[inputs.length];
        for(int i = 0; i < inputs.length; i++) {
            if (inputs[i] >= 0) {
                outputs[i] = (double) BoundMath.log(1 + inputs[i]);
            } else {
                outputs[i] = (double) -BoundMath.log(1 - inputs[i]);
            }
        }
        return outputs;
    }

    @Override
    public double[] backward(double[] backInput) {
        double[] results = new double[backInput.length];
        for(int i = 0; i < results.length; i++) {
            if (this.lastInput[i] >= 0) {
                results[i] = 1 / (1 + this.lastInput[i]);
            } else {
                results[i] = 1 / (1 - this.lastInput[i]);
            }
        }
        return results;
    }
}
