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
public class BiasLayer implements Layer<Float, Float, Float, Float> {

    private float weight;

    @Override
    public int getOutDim() {
        return 1;
    }

    @Override
    public Float forward(Float input) {
        return weight;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.core.dtrain.wnd.Layer#backward(java.lang.Object)
     */
    @Override
    public Float backward(Float backInput, float sig) {
        // TODO compute gradients here
        return backInput * weight;
    }

}
