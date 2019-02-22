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
import java.util.List;

import ml.shifu.guagua.io.Bytable;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideDenseLayer implements Layer<float[], float[], float[], float[]>, WeightInitializable, Bytable {

    /**
     * [in] float array of weights
     */
    private float[] weights;

    /**
     * Gradients, using map for sparse updates
     */
    private float[] wGrads;

    /**
     * # of inputs
     */
    private int in;

    /**
     * L2 level regularization parameter.
     */
    private float l2reg;

    /**
     * Last input used in backward computation
     */
    private float[] lastInput;

    /**
     * Columns of IDs
     */
    private List<Integer> columnIds;

    public WideDenseLayer(List<Integer> columnIds, float[] weights, int in, float l2reg) {
        this.weights = weights;
        this.in = in;
        this.setColumnIds(columnIds);
        this.l2reg = l2reg;
    }

    public WideDenseLayer(List<Integer> columnIds, int in, float l2reg) {
        this.in = in;
        this.setColumnIds(columnIds);
        this.weights = new float[in];
        this.l2reg = l2reg;
    }

    @Override
    public float[] forward(float[] inputs) {
        this.lastInput = inputs;
        float[] results = new float[1];
        for(int i = 0; i < inputs.length; i++) {
            results[0] += inputs[i] * this.weights[i];
        }
        return results;
    }

    @Override
    public float[] backward(float[] backInputs, float sig) {
        // gradients compute and L2 reg here
        for(int i = 0; i < this.in; i++) {
            this.wGrads[i] += (this.lastInput[i] * backInputs[0] * sig); // basic derivatives
            this.wGrads[i] += (this.l2reg * this.weights[i] * sig);// l2 loss derivatives
        }
        // no need compute backward outputs as it is last layer
        return null;
    }

    public void initGrads() {
        this.wGrads = new float[this.in];
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

    @Override
    public void initWeight(String policy) {
        // TODO
    }

    /**
     * @return the columnIds
     */
    public List<Integer> getColumnIds() {
        return columnIds;
    }

    /**
     * @param columnIds
     *            the columnIds to set
     */
    public void setColumnIds(List<Integer> columnIds) {
        this.columnIds = columnIds;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        // TODO Auto-generated method stub
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        // TODO Auto-generated method stub
    }
}
