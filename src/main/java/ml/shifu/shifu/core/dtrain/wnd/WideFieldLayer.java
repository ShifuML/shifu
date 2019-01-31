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
public class WideFieldLayer implements Layer<SparseInput, float[], float[], float[]> {

    private float[] weights;

    private int in;

    private int columnId;

    public WideFieldLayer(int columnId, float[] weights, int in) {
        this.weights = weights;
        this.in = in;
        this.setColumnId(columnId);
    }

    public WideFieldLayer(int columnId, int in) {
        this.in = in;
        this.setColumnId(columnId);
        // TODO init weights
    }

    @Override
    public float[] forward(SparseInput si) {
        int valueIndex = si.getValueIndex();
        return new float[] { this.weights[valueIndex] };
    }

    @Override
    public float[] backward(float[] backInputs) {
        assert backInputs.length == 1;
        float error = backInputs[0];

        float[] results = new float[this.weights.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = this.weights[i] * error;
        }
        // TODO sparse version backward wide layer major for gradients here, up backward could be ignored
        return results;
    }

    @Override
    public int getOutDim() {
        return 1;
    }

    /**
     * @return the weights
     */
    public float[] getWeights() {
        return weights;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(float[] weights) {
        this.weights = weights;
    }

    /**
     * @return the in
     */
    public int getIn() {
        return in;
    }

    /**
     * @param in
     *            the in to set
     */
    public void setIn(int in) {
        this.in = in;
    }

    /**
     * @return the columnId
     */
    public int getColumnId() {
        return columnId;
    }

    /**
     * @param columnId
     *            the columnId to set
     */
    public void setColumnId(int columnId) {
        this.columnId = columnId;
    }

}
