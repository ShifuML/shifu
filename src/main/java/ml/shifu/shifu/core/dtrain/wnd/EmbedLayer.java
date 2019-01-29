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

    private int in;
    
    private int columnId;
    
    public EmbedLayer(int columnId, float[][] weights, int out, int in) {
        this.setColumnId(columnId);
        this.setWeights(weights);
        this.setOut(out);
        this.setIn(in);
    }

    public EmbedLayer(int columnId, int out, int in) {
        this.setColumnId(columnId);
        this.setOut(out);
        this.setIn(in);
        // TODO init weights
    }

    @Override
    public int getOutDim() {
        return this.getOut();
    }

    @Override
    public float[] forward(SparseInput si) {
        int valueIndex = si.getValueIndex();
        return this.getWeights()[valueIndex];
    }

    @Override
    public float[] backward(float[] backInputs) {
        float[] results = new float[backInputs.length];
        for(int i = 0; i < results.length; i++) {
            for(int j = 0; j < backInputs.length; j++) {
                results[i] += this.getWeights()[i][j] * backInputs[j];
            }
        }
        return results;
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
     * @return the out
     */
    public int getOut() {
        return out;
    }

    /**
     * @param out
     *            the out to set
     */
    public void setOut(int out) {
        this.out = out;
    }

    /**
     * @return the weights
     */
    public float[][] getWeights() {
        return weights;
    }

    /**
     * @param weights the weights to set
     */
    public void setWeights(float[][] weights) {
        this.weights = weights;
    }

    /**
     * @return the columnId
     */
    public int getColumnId() {
        return columnId;
    }

    /**
     * @param columnId the columnId to set
     */
    public void setColumnId(int columnId) {
        this.columnId = columnId;
    }

}
