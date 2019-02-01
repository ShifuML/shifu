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
package ml.shifu.shifu.core.dtrain.wnd;

/**
 * @author pengzhang
 *
 */
public class ReLU extends Activiation {

    private float[] lastInput;

    @Override
    public float[] forward(float[] inputs) {
        this.lastInput = inputs;
        float[] outputs = new float[inputs.length];
        for(int i = 0; i < inputs.length; i++) {
            outputs[i] = Math.max(0, inputs[i]);
        }
        return outputs;
    }

    @Override
    public float[] backward(float[] outputs, float sig) {
        float[] results = new float[outputs.length];
        for(int i = 0; i < outputs.length; i++) {
            results[i] = this.lastInput[i] > 0 ? outputs[i] * 1f : 0f;
        }
        return results;
    }

}
