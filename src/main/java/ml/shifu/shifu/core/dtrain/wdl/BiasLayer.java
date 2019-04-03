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

/**
 * {@link BiasLayer} used in wide part of WideAndDeep architecture.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class BiasLayer extends AbstractLayer<Float, Float, Float, Float, BiasLayer> implements WeightInitializer<BiasLayer> {

    private float weight;

    private float wGrad;

    public BiasLayer(float weight) {
        this.weight = weight;
    }

    public BiasLayer() {
    }

    @Override
    public int getOutDim() {
        return 1;
    }

    @Override
    public Float forward(Float input) {
        return weight;
    }

    @Override
    public Float backward(Float backInput, float sig) {
        // no need l2 reg in bias layer
        this.wGrad = backInput * sig;
        // no need backward output computation as it is last layer.
        return backInput * weight;
    }

    public float getWeight() {
        return weight;
    }

    public void setWeight(float weight) {
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
    public float getwGrad() {
        return wGrad;
    }

    /**
     * @param wGrad
     *            the wGrad to set
     */
    public void setwGrad(float wGrad) {
        this.wGrad = wGrad;
    }

    public void initGrads() {
        this.wGrad = 0f;
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
                out.writeFloat(weight);
                break;
            case GRADIENTS:
                out.writeFloat(this.wGrad);
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
                this.weight = in.readFloat();
                break;
            case GRADIENTS:
                this.wGrad = in.readFloat();
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
    public void update(BiasLayer gradLayer, Optimizer optimizer) {
        this.weight -= optimizer.updateWeight(gradLayer.getwGrad(), getClass().getSimpleName());
    }

}
