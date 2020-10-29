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
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.layer.optimization.Optimizer;
import ml.shifu.shifu.core.dtrain.layer.optimization.PropOptimizer;
import ml.shifu.shifu.core.dtrain.layer.optimization.WeightOptimizer;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideDenseLayer extends AbstractLayer<double[], double[], double[], double[], WideDenseLayer>
        implements WeightInitializer<WideDenseLayer>, PropOptimizer<WideDenseLayer> {
    @SuppressWarnings("unused")
    private static final Logger LOG = LoggerFactory.getLogger(WideDenseLayer.class);
    /**
     * [in] double array of weights
     */
    private double[] weights;

    /**
     * Gradients, using map for sparse updates
     */
    private double[] wGrads;

    private WeightOptimizer optimizer;

    /**
     * # of inputs
     */
    private int in;

    /**
     * L2 level regularization parameter.
     */
    private double l2reg;   

    /**
     * Last input used in backward computation
     */
    private double[] lastInput;

    /**
     * Columns of IDs
     */
    private List<Integer> columnIds;

    private boolean isDebug = false;

    public WideDenseLayer() {
    }

    public WideDenseLayer(List<Integer> columnIds, double[] weights, int in, double l2reg) {
        this.weights = weights;
        this.in = in;
        this.columnIds = columnIds;
        this.l2reg = l2reg;
    }

    public WideDenseLayer(List<Integer> columnIds, int in, double l2reg) {
        this.in = in;
        this.setColumnIds(columnIds);
        this.weights = new double[in];
        this.l2reg = l2reg;
    }

    @Override
    public double[] forward(double[] inputs) {
        // if(this.isDebug) {
        // LOG.info("WideDenseLayer weights:" + Arrays.toString(this.weights));
        // LOG.info("WideDenseLayer inputs:" + Arrays.toString(inputs));
        // }
        this.lastInput = inputs;
        double[] results = new double[1];
        for(int i = 0; i < inputs.length; i++) {
            // LOG.debug("inputs[i]=" + inputs[i] + "this.weights[i]=" + this.weights[i]);
            results[0] += inputs[i] * this.weights[i];
        }

        // if(this.isDebug) {
        // LOG.info("WideDenseLayer results:" + Arrays.toString(results));
        // }
        return results;
    }

    @Override
    public double[] backward(double[] backInputs) {
        // gradients compute and L2 reg here
        for(int i = 0; i < this.in; i++) {
            // basic derivatives
            this.wGrads[i] += (this.lastInput[i] * backInputs[0]);
            // l2 loss derivatives
            this.wGrads[i] += (this.l2reg * backInputs[0]);
        }

        // if(this.isDebug()) {
        // LOG.info(" Training l2reg {}, backInputs {}, last input {}.", l2reg, backInputs[0],
        // Arrays.toString(lastInput));
        // LOG.info(" Training dense wGradients: {}.", Arrays.toString(wGrads));
        // }

        // no need compute backward outputs as it is last layer
        return null;
    }

    public void initGrads() {
        this.wGrads = new double[this.in];
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
     * @return the wGrads
     */
    public double[] getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(double[] wGrads) {
        this.wGrads = wGrads;
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
        out.writeDouble(this.l2reg);
        out.writeInt(this.in);

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                SerializationUtil.writeDoubleArray(out, this.weights, this.in);
                break;
            case GRADIENTS:
                SerializationUtil.writeDoubleArray(out, this.wGrads, this.in);
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
        this.l2reg = in.readDouble();
        this.in = in.readInt();

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                this.weights = SerializationUtil.readDoubleArray(in, this.weights, this.in);
                break;
            case GRADIENTS:
                this.wGrads = SerializationUtil.readDoubleArray(in, this.wGrads, this.in);
                break;
            default:
                break;
        }
    }

    @Override
    public void initWeight(InitMethod method) {
        this.weights = method.getInitialisable().initWeight(this.in);
    }

    @Override
    public void initWeight(WideDenseLayer updateModel) {
        for(int i = 0; i < this.in; i++) {
            this.weights[i] = updateModel.getWeights()[i];
        }
    }

    @Override
    public WideDenseLayer combine(WideDenseLayer from) {
        double[] fromGrads = from.getwGrads();
        for(int i = 0; i < this.in; i++) {
            wGrads[i] += fromGrads[i];
        }
        return this;
    }

    @Override
    public void update(WideDenseLayer gradLayer, Optimizer optimizer, String uniqueKey, double trainCount) {
        optimizer.update(this.weights, gradLayer.getwGrads(), uniqueKey, trainCount);
    }

    /**
     * @return the isDebug
     */
    public boolean isDebug() {
        return isDebug;
    }

    /**
     * @param isDebug
     *            the isDebug to set
     */
    public void setDebug(boolean isDebug) {
        this.isDebug = isDebug;
    }

    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        this.optimizer = new WeightOptimizer(this.in, learningRate, algorithm, reg, rl, algorithm);
    }

    @Override
    public void optimizeWeight(double numTrainSize, int iteration, WideDenseLayer model) {
        this.optimizer.calculateWeights(this.weights, model.getwGrads(), iteration, numTrainSize);
    }
}
