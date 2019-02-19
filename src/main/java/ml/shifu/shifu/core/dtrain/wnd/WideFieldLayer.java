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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ml.shifu.guagua.io.Bytable;

/**
 * {@link WideFieldLayer} is wide part input of WideAndDeep architecture. Per each column a {@link WideFieldLayer}
 * instance and each instanced will be forwarded and backwarded accordingly.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideFieldLayer implements Layer<SparseInput, float[], float[], float[]>, WeightInitializable, Bytable {

    /**
     * [in] float array of weights
     */
    private float[] weights;

    /**
     * Gradients, using map for sparse updates
     */
    private Map<Integer, Float> wGrads;

    /**
     * # of inputs
     */
    private int in;

    /**
     * ColumnConfig#columnNum for features used in this wide field layer.
     */
    private int columnId;

    /**
     * L2 level regularization parameter.
     */
    private float l2reg;

    /**
     * Last input used in backward computation
     */
    private SparseInput lastInput;

    public WideFieldLayer(int columnId, float[] weights, int in, float l2reg) {
        this.weights = weights;
        this.in = in;
        this.columnId = columnId;
        this.l2reg = l2reg;
    }

    public WideFieldLayer(int columnId, int in, float l2reg) {
        this.in = in;
        this.columnId = columnId;
        this.weights = new float[in];
        this.l2reg = l2reg;
    }

    @Override
    public float[] forward(SparseInput si) {
        this.lastInput = si;
        int valueIndex = si.getValueIndex();
        return new float[] { si.getValue() * this.weights[valueIndex] };
    }

    @Override
    public float[] backward(float[] backInputs, float sig) {
        assert backInputs.length == 1;

        int valueIndex = this.lastInput.getValueIndex();
        Float grad = this.wGrads.get(valueIndex);
        float tmpGrad = grad == null ? 0 : grad;
        tmpGrad += (this.lastInput.getValue() * backInputs[0] * sig); // category value here is 1f
        tmpGrad += (this.lastInput.getValue() * this.l2reg * this.weights[valueIndex] * sig);// l2 loss
        this.wGrads.put(valueIndex, tmpGrad);

        // no need compute backward outputs as it is last layer
        return null;
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

    /**
     * @return the wGrads
     */
    public Map<Integer, Float> getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(Map<Integer, Float> wGrads) {
        this.wGrads = wGrads;
    }

    /**
     * @return the l2reg
     */
    public float getL2reg() {
        return l2reg;
    }

    /**
     * @param l2reg
     *            the l2reg to set
     */
    public void setL2reg(float l2reg) {
        this.l2reg = l2reg;
    }

    public void initGrads() {
        this.wGrads = new HashMap<>();
    }

    @Override
    public void initWeight(String policy) {
        // TODO
    }

    /* (non-Javadoc)
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        // TODO Auto-generated method stub
        
    }

    /* (non-Javadoc)
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        // TODO Auto-generated method stub
        
    }
}
