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
package ml.shifu.shifu.core.dtrain.wdl;

import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * {@link EmbedFieldLayer} is for each column like sparse categorical feature. The input of this layer is one-hot
 * encoding while the output is dense vector.
 * 
 * <p>
 * Inputs of EmbedLayer is typical sparse input and with this, forward/backward computation can be leveraged with far
 * less computation.
 * 
 * <p>
 * Bias is not supported as in embed with bias, sparse gradients will be missed.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class EmbedFieldLayer extends AbstractLayer<SparseInput, float[], float[], float[], EmbedFieldLayer>
        implements WeightInitializer<EmbedFieldLayer> {

    /**
     * [in, out] array for deep matrix weights
     */
    private float[][] weights;

    /**
     * Weight gradients in back computation
     */
    private Map<Integer, float[]> wGrads;

    /**
     * The output dimension
     */
    private int out;

    /**
     * The input dimension (bias not included)
     */
    private int in;

    /**
     * ColumnConfig#columnNum as id for such embedding layer.
     */
    private int columnId;

    /**
     * Last input used for backward gradients computation
     */
    private SparseInput lastInput;

    public EmbedFieldLayer() {
    }

    public EmbedFieldLayer(int columnId, float[][] weights, int out, int in) {
        this.columnId = columnId;
        this.weights = weights;
        this.out = out;
        this.in = in;
    }

    public EmbedFieldLayer(int columnId, int out, int in) {
        this.columnId = columnId;
        this.out = out;
        this.in = in;
        this.weights = new float[in][out];
    }

    @Override
    public int getOutDim() {
        return this.getOut();
    }

    @Override
    public float[] forward(SparseInput si) {
        this.lastInput = si;
        int valueIndex = si.getValueIndex();
        float[] results = new float[this.out];
        for(int i = 0; i < results.length; i++) {
            results[i] = si.getValue() * this.getWeights()[valueIndex][i];
        }
        return results;
    }

    @Override
    public float[] backward(float[] backInputs) {
        // gradients computation
        int valueIndex = this.lastInput.getValueIndex();
        this.wGrads.computeIfAbsent(valueIndex, k -> new float[this.out]);
        for(int j = 0; j < this.out; j++) {
            this.wGrads.get(valueIndex)[j] += (this.lastInput.getValue() * backInputs[j]);
        }

        // no need compute backward outputs as it is last layer
        return null;
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
     * @param weights
     *            the weights to set
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
     * @param columnId
     *            the columnId to set
     */
    public void setColumnId(int columnId) {
        this.columnId = columnId;
    }

    /**
     * @return the wGrads
     */
    public Map<Integer, float[]> getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(Map<Integer, float[]> wGrads) {
        this.wGrads = wGrads;
    }

    public void initGrads() {
        this.wGrads = new HashMap<>();
    }

    @Override
    public void initWeight(InitMethod method) {
        this.weights = method.getInitialisable().initWeight(this.in, this.out);
    }

    @Override
    public void initWeight(EmbedFieldLayer updateModel) {
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                this.weights[i][j] = updateModel.getWeights()[i][j];
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.columnId);
        out.writeInt(this.in);
        out.writeInt(this.out);

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                SerializationUtil.write2DimFloatArray(out, this.weights, this.in, this.out);
                break;
            case GRADIENTS:
                if(this.wGrads == null) {
                    out.writeInt(0);
                } else {
                    out.writeInt(this.wGrads.size());
                    for(Entry<Integer, float[]> entry: this.wGrads.entrySet()) {
                        out.writeInt(entry.getKey());
                        SerializationUtil.writeFloatArray(out, entry.getValue(), this.out);
                    }
                }
                break;
            default:
                break;
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        this.columnId = in.readInt();
        this.in = in.readInt();
        this.out = in.readInt();

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                this.weights = SerializationUtil.read2DimFloatArray(in, this.weights, this.in, this.out);
                break;
            case GRADIENTS:
                if(this.wGrads != null) {
                    this.wGrads.clear();
                } else {
                    this.wGrads = new HashMap<Integer, float[]>();
                }
                int gradSize = in.readInt();
                for(int i = 0; i < gradSize; i++) {
                    int lineNumber = in.readInt();
                    float[] grad = SerializationUtil.readFloatArray(in, null, this.out);
                    this.wGrads.put(lineNumber, grad);
                }
                break;
            default:
                break;
        }
    }

    @Override
    public EmbedFieldLayer combine(EmbedFieldLayer from) {
        if(columnId != from.getColumnId()) {
            return this;
        }
        Map<Integer, float[]> fromGrads = from.getwGrads();
        for(Entry<Integer, float[]> entry: fromGrads.entrySet()) {
            Integer index = entry.getKey();
            float[] grad = entry.getValue();
            if(wGrads.containsKey(index)) {
                float[] thisGrad = wGrads.get(index);
                for(int i = 0; i < this.out; i++) {
                    grad[i] += thisGrad[i];
                }
            }
            wGrads.put(index, grad);
        }
        return this;
    }

    @Override
    public void update(EmbedFieldLayer gradLayer, Optimizer optimizer) {
        optimizer.batchUpdate(this.weights, gradLayer.getwGrads());
    }
}
