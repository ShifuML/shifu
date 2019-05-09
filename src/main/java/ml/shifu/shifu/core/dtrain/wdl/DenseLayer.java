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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * {@link DenseLayer} defines normal hidden layer in neural network while activation is not included but in one
 * specified layer.
 *
 * As common dense layer, bias part is included.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DenseLayer extends AbstractLayer<float[], float[], float[], float[], DenseLayer>
        implements WeightInitializer<DenseLayer> {
    private static final Logger LOG = LoggerFactory.getLogger(DenseLayer.class);

    /**
     * [in, out] array for deep matrix weights
     */
    private float[][] weights;

    /**
     * Weight gradients in back computation
     */
    private float[][] wGrads;

    /**
     * [out] array for bias in input of such DenseLayer
     */
    private float[] bias;

    /**
     * Bias gradients in back computation
     */
    private float[] bGrads;

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
    private float l2reg;

    /**
     * Layer inputs used for backward gradients computation, tmp use for computation
     */
    private float[] lastInput = null;

    public DenseLayer() {
    }

    public DenseLayer(float[][] weights, float[] bias, int out, int in, float l2reg) {
        this.weights = weights;
        this.bias = bias;
        this.out = out;
        this.in = in;
        this.l2reg = l2reg;
    }

    public DenseLayer(int out, int in, float l2reg) {
        this.out = out;
        this.in = in;
        this.l2reg = l2reg;
        this.bias = new float[out];
        this.weights = new float[in][out];
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
     * @return the bias
     */
    public float[] getBias() {
        return bias;
    }

    /**
     * @param bias
     *            the bias to set
     */
    public void setBias(float[] bias) {
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
    public float[] forward(float[] inputs) {
        this.lastInput = inputs;
        float[] results = new float[this.out];
        for(int i = 0; i < results.length; i++) {
            for(int j = 0; j < inputs.length; j++) {
                results[i] += inputs[j] * this.weights[j][i];
            }
            results[i] += bias[i];
        }
        return results;
    }

    @Override
    public float[] backward(float[] backInputs) {
        // gradients compute and L2 reg here
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                this.wGrads[i][j] += (this.lastInput[i] * backInputs[j]); // basic derivatives
                this.wGrads[i][j] += (this.l2reg * this.weights[i][j]);// l2 loss derivatives
            }
        }
        for(int j = 0; j < this.out; j++) {
            this.bGrads[j] = (backInputs[j]); // no need l2 reg here as bias no need
        }

        // compute back inputs
        float[] results = new float[this.in];
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
            this.wGrads = new float[this.in][this.out];
        }
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                this.wGrads[i][j] = 0f;
            }
        }

        if(this.bGrads == null) {
            // reuse same array
            this.bGrads = new float[this.bias.length];
        }
        for(int j = 0; j < this.out; j++) {
            this.bGrads[j] = 0f;
        }
    }

    /**
     * @return the wGrads
     */
    public float[][] getwGrads() {
        return wGrads;
    }

    /**
     * @param wGrads
     *            the wGrads to set
     */
    public void setwGrads(float[][] wGrads) {
        this.wGrads = wGrads;
    }

    /**
     * @return the bGrads
     */
    public float[] getbGrads() {
        return bGrads;
    }

    /**
     * @param bGrads
     *            the bGrads to set
     */
    public void setbGrads(float[] bGrads) {
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
        out.writeFloat(this.l2reg);
        out.writeInt(this.in);
        out.writeInt(this.out);

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                SerializationUtil.write2DimFloatArray(out, this.weights, this.in, this.out);
                SerializationUtil.writeFloatArray(out, this.bias, this.out);
                break;
            case GRADIENTS:
                SerializationUtil.write2DimFloatArray(out, this.wGrads, this.in, this.out);
                SerializationUtil.writeFloatArray(out, this.bGrads, this.out);
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
        this.l2reg = in.readFloat();
        this.in = in.readInt();
        this.out = in.readInt();

        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                this.weights = SerializationUtil.read2DimFloatArray(in, this.weights, this.in, this.out);
                this.bias = SerializationUtil.readFloatArray(in, this.bias, this.out);
                break;
            case GRADIENTS:
                this.wGrads = SerializationUtil.read2DimFloatArray(in, this.wGrads, this.in, this.out);
                this.bGrads = SerializationUtil.readFloatArray(in, this.bGrads, this.out);
                break;
            default:
                break;
        }
    }

    @Override
    public DenseLayer combine(DenseLayer from) {
        float[][] fromWGrads = from.getwGrads();
        for(int i = 0; i < this.in; i++) {
            for(int j = 0; j < this.out; j++) {
                wGrads[i][j] += fromWGrads[i][j];
            }
        }
        float[] fromBGrads = from.getbGrads();
        for(int i = 0; i < this.out; i++) {
            bGrads[i] += fromBGrads[i];
        }
        return this;
    }

    @Override
    public void update(DenseLayer gradLayer, Optimizer optimizer, String uniqueKey) {
        LOG.error("Before update: weights" + Arrays.deepToString(this.weights));
        optimizer.batchUpdate(this.weights, gradLayer.getwGrads(), uniqueKey);
        LOG.error("After update: weights" + Arrays.deepToString(this.weights));
        optimizer.update(this.bias, gradLayer.getbGrads(), uniqueKey);
    }
}
