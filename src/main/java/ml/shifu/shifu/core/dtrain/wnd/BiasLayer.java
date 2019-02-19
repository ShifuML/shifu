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

import ml.shifu.guagua.io.Bytable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * {@link BiasLayer} used in wide part of WideAndDeep architecture.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class BiasLayer implements Layer<Float, Float, Float, Float>, WeightInitialisable, Bytable {

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
        this.wGrad = backInput * sig; // no need l2 reg in bias layer
        // no need backward output computation as it is last layer.
        return backInput * weight;
    }

    public float getWeight() {
        return weight;
    }

    public void setWeight(float weight) {
        this.weight = weight;
    }

    @Override public void initWeight(InitMethod method) {
        this.weight = method.getInitialisable().initWeight();
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
