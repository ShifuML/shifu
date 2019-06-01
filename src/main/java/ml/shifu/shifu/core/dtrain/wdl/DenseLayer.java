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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;

/**
 * {@link DenseLayer} defines normal hidden layer in neural network while activation is not included but in one
 * specified layer.
 *
 * As common dense layer, bias part is included.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DenseLayer extends AbstractLayer<double[], double[], double[], double[], DenseLayer>
        implements WeightInitializer<DenseLayer> {
    @SuppressWarnings("unused")
    private static final Logger LOG = LoggerFactory.getLogger(DenseLayer.class);

    /**
     * [in, out] array for deep matrix weights
     */
    private double[][] weights;

    /**
     * Weight gradients in back computation
     */
    private double[][] wGrads;

    /**
     * [out] array for bias in input of such DenseLayer
     */
    private double[] bias;

    /**
     * Bias gradients in back computation
     */
    private double[] bGrads;

    /**
     * The output dimension
     */
    private int out;

    /**
     * The input dimension (bias not included)
     */
    private int in;

    /**
     * L2 level regularization parameter.
     */
    private double l2reg;

    /**
     * Layer inputs used for backward gradients computation, tmp use for computation
     */
    private double[] lastInput = null;

    public DenseLayer() {
    }

    public DenseLayer(double[][] weights, double[] bias, int out, int in, double l2reg) {
        this.weights = weights;
        this.bias = bias;
        this.out = out;
        this.in = in;
        this.l2reg = l2reg;
    }

    public DenseLayer(int out, int in, double l2reg) {
        this.out = out;
        this.in = in;
        this.l2reg = l2reg;
        this.bias = new double[out];
        this.weights = new double[in][out];
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
     * @return the bias
     */
    public double[] getBias() {
        return bias;
    }

    /**
     * @param bias
     *            the bias to set
     */
    public void setBias(double[] bias) {
        this.bias = bias;
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

    @Override
    public int getOutDim() {
        return this.out;
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

    @Override
    public double[] forward(double[] inputs) {
        this.lastInput = inputs;
        double[] results = new double[this.out];
        for(int i = 0; i < results.length; i++) {
            for(int j = 0; j < inputs.length; j++) {
                results[i] += inputs[j] * this.weights[j][i];
            }
            results[i] += bias[i];
        }
        return results;
    }

    @Override
    public double[] backward(double[] backInputs) {
        // gradients compute and L2 reg here
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                // basic derivatives
                this.wGrads[i][j] += (this.lastInput[i] * backInputs[j]);
                // l2 loss derivatives
//                this.wGrads[i][j] += (this.l2reg * backInputs[j]);
            }
        }
        for(int j = 0; j < this.out; j++) {
            // no need l2 reg here as bias no need
            this.bGrads[j] = (backInputs[j]);
        }

        // compute back inputs
        double[] results = new double[this.in];
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < backInputs.length; j++) {
                results[i] += (backInputs[j] * this.weights[i][j]);
            }
        }
        return results;
    }

    public void initGrads() {
        if(this.wGrads == null) {
            // reuse same array
            this.wGrads = new double[this.in][this.out];
        }
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                this.wGrads[i][j] = 0f;
            }
        }

        if(this.bGrads == null) {
            // reuse same array
            this.bGrads = new double[this.bias.length];
        }
        for(int j = 0; j < this.out; j++) {
            this.bGrads[j] = 0f;
        }
    }

    /**
     * @return the wGrads
     */
    public double[][] getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(double[][] wGrads) {
        this.wGrads = wGrads;
    }

    /**
     * @return the bGrads
     */
    public double[] getbGrads() {
        return bGrads;
    }

    /**
     * @param bGrads
     *            the bGrads to set
     */
    public void setbGrads(double[] bGrads) {
        this.bGrads = bGrads;
    }

    @Override
    public void initWeight(InitMethod method) {
        this.weights = method.getInitialisable().initWeight(this.in, this.out);
    }

    @Override
    public void initWeight(DenseLayer updateModel) {
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
        out.writeDouble(this.l2reg);
        out.writeInt(this.in);
        out.writeInt(this.out);

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                SerializationUtil.write2DimDoubleArray(out, this.weights, this.in, this.out);
                SerializationUtil.writeDoubleArray(out, this.bias, this.out);
                break;
            case GRADIENTS:
                SerializationUtil.write2DimDoubleArray(out, this.wGrads, this.in, this.out);
                SerializationUtil.writeDoubleArray(out, this.bGrads, this.out);
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
        this.out = in.readInt();

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                this.weights = SerializationUtil.read2DimDoubleArray(in, this.weights, this.in, this.out);
                this.bias = SerializationUtil.readDoubleArray(in, this.bias, this.out);
                break;
            case GRADIENTS:
                this.wGrads = SerializationUtil.read2DimDoubleArray(in, this.wGrads, this.in, this.out);
                this.bGrads = SerializationUtil.readDoubleArray(in, this.bGrads, this.out);
                break;
            default:
                break;
        }
    }

    @Override
    public DenseLayer combine(DenseLayer from) {
        double[][] fromWGrads = from.getwGrads();
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                wGrads[i][j] += fromWGrads[i][j];
            }
        }
        double[] fromBGrads = from.getbGrads();
        for(int i = 0; i < this.out; i++) {
            bGrads[i] += fromBGrads[i];
        }
        return this;
    }

    @Override
    public void update(DenseLayer gradLayer, Optimizer optimizer, String uniqueKey, double trainCount) {
        optimizer.batchUpdate(this.weights, gradLayer.getwGrads(), uniqueKey, trainCount);
        optimizer.update(this.bias, gradLayer.getbGrads(), uniqueKey, trainCount);
    }
}
