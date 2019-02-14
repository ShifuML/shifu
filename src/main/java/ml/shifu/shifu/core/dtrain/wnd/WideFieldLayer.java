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
 * {@link WideFieldLayer} is wide part input of WideAndDeep architecture. Per each column a {@link WideFieldLayer}
 * instance and each instanced will be forwarded and backwarded accordingly.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideFieldLayer implements Layer<SparseInput, float[], float[], float[]>, WeightInitializable {

    /**
     * [in] float array of weights
     */
    private float[] weights;

    /**
     * # of inputs
     */
    private int in;

    /**
     * ColumnConfig#columnNum for features used in this wide field layer.
     */
    private int columnId;

    public WideFieldLayer(int columnId, float[] weights, int in) {
        this.weights = weights;
        this.in = in;
        this.columnId = columnId;
    }

    public WideFieldLayer(int columnId, int in) {
        this.in = in;
        this.columnId = columnId;
        this.weights = new float[in];
    }

    @Override
    public float[] forward(SparseInput si) {
        int valueIndex = si.getValueIndex();
        return new float[] { this.weights[valueIndex] };
    }

    @Override
    public float[] backward(float[] backInputs, float sig) {
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

    @Override
    public void initWeight(String policy) {
        //TODO
    }
}
