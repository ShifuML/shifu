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
package ml.shifu.shifu.core.dtrain.layer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.layer.optimization.Optimizer;
import ml.shifu.shifu.core.dtrain.layer.optimization.PropOptimizer;
import ml.shifu.shifu.core.dtrain.layer.optimization.WeightOptimizer;

/**
 * {@link WideFieldLayer} is wide part input of WideAndDeep architecture. Per each column a {@link WideFieldLayer}
 * instance and each instanced will be forwarded and backwarded accordingly.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideFieldLayer extends AbstractLayer<SparseInput, double[], double[], double[], WideFieldLayer>
        implements WeightInitializer<WideFieldLayer>, PropOptimizer<WideFieldLayer> {
    private static final Logger LOG = LoggerFactory.getLogger(WideFieldLayer.class);
    /**
     * [in] double array of weights
     */
    private double[] weights;

    /**
     * Weight optimizer
     */
    private WeightOptimizer optimizer;

    /**
     * Gradients, using map for sparse updates
     */
    private Map<Integer, Double> wGrads;

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
    private double l2reg;

    /**
     * Last input used in backward computation
     */
    private SparseInput lastInput;

    public WideFieldLayer() {
    }

    public WideFieldLayer(int columnId, double[] weights, int in, double l2reg) {
        this.weights = weights;
        this.in = in;
        this.columnId = columnId;
        this.l2reg = l2reg;
    }

    public WideFieldLayer(int columnId, int in, double l2reg) {
        this.in = in;
        this.columnId = columnId;
        this.weights = new double[in];
        this.l2reg = l2reg;
    }

    @Override
    public double[] forward(SparseInput si) {
        this.lastInput = si;
        int valueIndex = si.getValueIndex();
        if(valueIndex < this.weights.length && valueIndex >= 0) {
            return new double[] { si.getValue() * this.weights[valueIndex] };
        }
        return new double[] { 0d };
    }

    @Override
    public double[] backward(double[] backInputs) {
        assert backInputs.length == 1;

        int valueIndex = this.lastInput.getValueIndex();
        Double grad = this.wGrads.get(valueIndex);
        double tmpGrad = grad == null ? 0 : grad;
        // category value here is 1f
        tmpGrad += (this.lastInput.getValue() * backInputs[0]);
        // l2 loss
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
    public double[] getWeights() {
        return weights;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(double[] weights) {
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
    public Map<Integer, Double> getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(Map<Integer, Double> wGrads) {
        this.wGrads = wGrads;
    }

    /**
     * @return the l2reg
     */
    public double getL2reg() {
        return l2reg;
    }

    /**
     * @param l2reg
     *            the l2reg to set
     */
    public void setL2reg(double l2reg) {
        this.l2reg = l2reg;
    }

    public void initGrads() {
        this.wGrads = new HashMap<Integer, Double>();
    }

    @Override
    public void initWeight(InitMethod method) {
        this.weights = method.getInitialisable().initWeight(this.in);
    }

    @Override
    public void initWeight(WideFieldLayer updateModel) {
        for(int i = 0; i < this.in; i++) {
            this.weights[i] = updateModel.getWeights()[i];
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
        out.writeDouble(this.l2reg);
        out.writeInt(this.in);

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                SerializationUtil.writeDoubleArray(out, this.weights, this.in);
                break;
            case GRADIENTS:
                if(this.wGrads == null) {
                    out.writeInt(0);
                } else {
                    out.writeInt(this.wGrads.size());
                    for(Entry<Integer, Double> entry: this.wGrads.entrySet()) {
                        out.writeInt(entry.getKey());
                        out.writeDouble(entry.getValue());
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
        this.l2reg = in.readDouble();
        this.in = in.readInt();

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                this.weights = SerializationUtil.readDoubleArray(in, this.weights, this.in);
                break;
            case GRADIENTS:
                if(this.wGrads != null) {
                    this.wGrads.clear();
                } else {
                    this.wGrads = new HashMap<Integer, Double>();
                }
                int size = in.readInt();
                for(int i = 0; i < size; i++) {
                    this.wGrads.put(in.readInt(), in.readDouble());
                }
                break;
            default:
                break;
        }
    }

    @Override
    public WideFieldLayer combine(WideFieldLayer from) {
        if(columnId != from.getColumnId()) {
            return this;
        }
        Map<Integer, Double> fromGrads = from.getwGrads();
        for(Entry<Integer, Double> entry: fromGrads.entrySet()) {
            Integer index = entry.getKey();
            double grad = entry.getValue();
            wGrads.put(index, grad + wGrads.getOrDefault(index, 0.0d));
        }
        return this;
    }

    @Override
    public void update(WideFieldLayer gradLayer, Optimizer optimizer, String uniqueKey, double trainCount) {
        LOG.info("Column {}, length {}, gradients {}.", this.columnId, this.weights.length, gradLayer.getwGrads());
        optimizer.update(this.weights, gradLayer.getwGrads(), uniqueKey, trainCount);
    }

    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        this.optimizer = new WeightOptimizer(this.in, learningRate, algorithm, reg, rl, algorithm);
    }

    @Override
    public void optimizeWeight(double numTrainSize, int iteration, WideFieldLayer model) {
        for(Map.Entry<Integer, Double> entry: model.getwGrads().entrySet()) {
            int index = entry.getKey();
            if(index < this.in) {
                this.optimizer.calculateWeights(this.weights, index, entry.getValue(), numTrainSize);
            } else {
                LOG.error("index {} in EmbedFieldLayer gradient great than in {}", index, this.in);
            }
        }
    }
}
