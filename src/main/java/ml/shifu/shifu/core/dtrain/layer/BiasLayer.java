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

import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.layer.optimization.Optimizer;
import ml.shifu.shifu.core.dtrain.layer.optimization.WeightOptimizer;

/**
 * {@link BiasLayer} used in wide part of WideAndDeep architecture.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class BiasLayer extends AbstractLayer<Double, Double, Double, Double, BiasLayer>
        implements WeightInitializer<BiasLayer> {

    private double weight;

    private double wGrad;

    private WeightOptimizer optimizer;

    public BiasLayer(double weight) {
        this.weight = weight;
    }

    public BiasLayer() {
    }

    @Override
    public int getOutDim() {
        return 1;
    }

    @Override
    public Double forward(Double input) {
        return weight;
    }

    @Override
    public Double backward(Double backInput) {
        // no need l2 reg in bias layer
        this.wGrad = backInput;
        // no need backward output computation as it is last layer.
        return backInput * weight;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    @Override
    public void initWeight(InitMethod method) {
        this.weight = method.getInitialisable().initWeight();
    }

    @Override
    public void initWeight(BiasLayer updateModel) {
        this.weight = updateModel.getWeight();
    }

    /**
     * @return the wGrad
     */
    public double getwGrad() {
        return wGrad;
    }

    /**
     * @param wGrad
     *            the wGrad to set
     */
    public void setwGrad(double wGrad) {
        this.wGrad = wGrad;
    }

    public void initGrads() {
        this.wGrad = 0f;
    }

    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        this.optimizer = new WeightOptimizer(1, learningRate, algorithm, reg, rl, algorithm);
    }

    public void optimizeWeight(double numTrainSize, int iteration, BiasLayer model) {
        double[] newWgt = this.optimizer.calculateWeights(new double[] { this.weight },
                new double[] { model.getwGrad() }, iteration, numTrainSize);
        this.weight = newWgt[0];
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                out.writeDouble(weight);
                break;
            case GRADIENTS:
                out.writeDouble(this.wGrad);
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
        switch(this.serializationType) {
            case WEIGHTS:
            case MODEL_SPEC:
                this.weight = in.readDouble();
                break;
            case GRADIENTS:
                this.wGrad = in.readDouble();
                break;
            default:
                break;
        }
    }

    @Override
    public BiasLayer combine(BiasLayer from) {
        this.wGrad += from.getwGrad();
        return this;
    }

    @Override
    public void update(BiasLayer gradLayer, Optimizer optimizer, String uniqueKey, double trainCount) {
        this.weight -= optimizer.updateWeight(gradLayer.getwGrad(), uniqueKey, trainCount);
    }

}
