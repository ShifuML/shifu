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
 * TODO
 * 
 * @author pengzhang
 */
public class EmbedLayer implements Layer<SparseInput, float[], float[], float[]> {

    private float[][] weights;

    private int out;

    @Override
    public int getOutDim() {
        return this.out;
    }

    @Override
    public float[] forward(SparseInput si) {
        int valueIndex = si.getValueIndex();
        return this.weights[valueIndex];
    }

    @Override
    public float[] backward(float[] backInputs) {
        float[] results = new float[backInputs.length];
        for(int i = 0; i < results.length; i++) {
            for(int j = 0; j < backInputs.length; j++) {
                results[i] += this.weights[i][j] * backInputs[j];
            }
        }
        return results;
    }

}
