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

import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.wdl.optimization.PropOptimizer;
import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;
import ml.shifu.shifu.core.dtrain.wdl.optimization.WeightOptimizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
public class EmbedFieldLayer extends AbstractLayer<SparseInput, double[], double[], double[], EmbedFieldLayer>
        implements WeightInitializer<EmbedFieldLayer>, PropOptimizer<EmbedFieldLayer> {
    private static final Logger LOG = LoggerFactory.getLogger(EmbedFieldLayer.class);

    /**
     * [in, out] array for deep matrix weights
     */
    private double[][] weights;

    /**
     * [in] array weight optimizers
     */
    private WeightOptimizer[] optimizers;

    /**
     * Weight gradients in back computation
     */
    private Map<Integer, double[]> wGrads;

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

    public EmbedFieldLayer(int columnId, double[][] weights, int out, int in) {
        this.columnId = columnId;
        this.weights = weights;
        this.out = out;
        this.in = in;
    }

    public EmbedFieldLayer(int columnId, int out, int in) {
        this.columnId = columnId;
        this.out = out;
        this.in = in;
        this.weights = new double[in][out];
    }

    @Override
    public int getOutDim() {
        return this.getOut();
    }

    @Override
    public double[] forward(SparseInput si) {
        this.lastInput = si;
        int valueIndex = si.getValueIndex();
        double[] results = new double[this.out];
        if(valueIndex < weights.length && valueIndex >= 0) {
            for(int i = 0; i < results.length; i++) {
                results[i] = si.getValue() * this.getWeights()[valueIndex][i];
            }
        } else {
            LOG.error("valueIndex=" + valueIndex + ", columnId=" + columnId + ", in=" + in + ", out=" + out);
        }
        return results;
    }

    @Override
    public double[] backward(double[] backInputs) {
        // gradients computation
        int valueIndex = this.lastInput.getValueIndex();
        this.wGrads.computeIfAbsent(valueIndex, k -> new double[this.out]);
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
    public double[][] getWeights() {
        return weights;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(double[][] weights) {
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
    public Map<Integer, double[]> getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(Map<Integer, double[]> wGrads) {
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
                SerializationUtil.write2DimDoubleArray(out, this.weights, this.in, this.out);
                break;
            case GRADIENTS:
                if(this.wGrads == null) {
                    out.writeInt(0);
                } else {
                    out.writeInt(this.wGrads.size());
                    for(Entry<Integer, double[]> entry: this.wGrads.entrySet()) {
                        out.writeInt(entry.getKey());
                        SerializationUtil.writeDoubleArray(out, entry.getValue(), this.out);
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
                this.weights = SerializationUtil.read2DimDoubleArray(in, this.weights, this.in, this.out);
                break;
            case GRADIENTS:
                if(this.wGrads != null) {
                    this.wGrads.clear();
                } else {
                    this.wGrads = new HashMap<Integer, double[]>();
                }
                int gradSize = in.readInt();
                for(int i = 0; i < gradSize; i++) {
                    int lineNumber = in.readInt();
                    double[] grad = SerializationUtil.readDoubleArray(in, null, this.out);
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
        Map<Integer, double[]> fromGrads = from.getwGrads();
        for(Entry<Integer, double[]> entry: fromGrads.entrySet()) {
            Integer index = entry.getKey();
            double[] grad = entry.getValue();
            if(wGrads.containsKey(index)) {
                double[] thisGrad = wGrads.get(index);
                for(int i = 0; i < this.out; i++) {
                    grad[i] += thisGrad[i];
                }
            }
            wGrads.put(index, grad);
        }
        return this;
    }

    @Override
    public void update(EmbedFieldLayer gradLayer, Optimizer optimizer, String uniqueKey, double trainCount) {
        optimizer.batchUpdate(this.weights, gradLayer.getwGrads(), uniqueKey, trainCount);
    }

    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        this.optimizers = new WeightOptimizer[this.in];
        for(int i = 0; i < this.in; i++) {
            this.optimizers[i] = new WeightOptimizer(this.out, learningRate, algorithm, reg, rl);
        }
    }

    @Override
    public void optimizeWeight(double numTrainSize, int iteration, EmbedFieldLayer model) {
        for(Map.Entry<Integer, double[]> entry: model.getwGrads().entrySet()) {
            int index = entry.getKey();
            if(index < this.in) {
                this.optimizers[index].calculateWeights(this.weights[index], entry.getValue(), iteration, numTrainSize);
            } else {
                LOG.error("index {} in EmbedFieldLayer gradient great than in {}", index, this.in);
            }
        }
    }
}
