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
public class Sigmoid extends Activiation {

    private float[] lastForward;

    @Override
    public float[] forward(float[] in) {
        assert in != null;

        float[] results = new float[in.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = (float) (1 / (1 + Math.min(1.0E19, Math.exp(-20 * in[i]))));
        }

        // temp saved for backward usage
        this.lastForward = results;
        return results;
    }

    @Override
    public float[] backward(float[] out) {
        assert out != null;
        assert out.length == lastForward.length;

        float[] results = new float[out.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = out[i] * lastForward[i] * (1f - lastForward[i]);
        }
        return results;
    }

}
