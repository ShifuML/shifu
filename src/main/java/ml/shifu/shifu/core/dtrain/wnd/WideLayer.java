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
import ml.shifu.shifu.core.dtrain.AssertUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link WideLayer} defines wide part of WideAndDeep. It includes a list of {@link WideFieldLayer} instances (each one
 * is for each wide column) and one {@link BiasLayer}.
 * 
 * <p>
 * {@link SparseInput} is leveraged for wide columns as with one-hot encoding to save more computation.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideLayer
        implements Layer<List<SparseInput>, float[], float[], List<float[]>>, WeightInitializer, Bytable {

    /**
     * Layers for all wide columns.
     */
    private List<WideFieldLayer> layers;

    /**
     * Bias layer
     */
    private BiasLayer bias;

    public WideLayer(List<WideFieldLayer> layers, BiasLayer bias) {
        this.layers = layers;
        this.bias = bias;
    }

    @Override
    public int getOutDim() {
        int len = 0;
        for(WideFieldLayer layer: getLayers()) {
            len += layer.getOutDim();
        }
        return len;
    }

    @Override
    public float[] forward(List<SparseInput> inputList) {
        AssertUtils.assertListNotNullAndSizeEqual(this.getLayers(), inputList);
        float[] results = new float[layers.get(0).getOutDim()];
        for(int i = 0; i < getLayers().size(); i++) {
            float[] fOuts = this.getLayers().get(i).forward(inputList.get(i));
            for(int j = 0; j < results.length; j++) {
                results[j] += fOuts[j];
            }
        }

        for(int j = 0; j < results.length; j++) {
            results[j] += bias.forward(1f);
        }

        return results;
    }

    @Override
    public List<float[]> backward(float[] backInputs, float sig) {
        // below backward call is for gradients computation in WideFieldLayer and BiasLayer
        List<float[]> list = new ArrayList<>();
        for(int i = 0; i < getLayers().size(); i++) {
            list.add(this.getLayers().get(i).backward(backInputs, sig));
        }
        list.add(new float[] { bias.backward(backInputs[0], sig) });
        return list;
    }

    /**
     * @return the layers
     */
    public List<WideFieldLayer> getLayers() {
        return layers;
    }

    /**
     * @param layers
     *            the layers to set
     */
    public void setLayers(List<WideFieldLayer> layers) {
        this.layers = layers;
    }

    /**
     * @return the bias
     */
    public BiasLayer getBias() {
        return bias;
    }

    /**
     * @param bias
     *            the bias to set
     */
    public void setBias(BiasLayer bias) {
        this.bias = bias;
    }

    @Override public void initWeight(InitMethod method) {
        for(WideFieldLayer layer: this.layers) {
            layer.initWeight(method);
        }
    }

    public void initGrads() {
        for(WideFieldLayer layer: this.layers) {
            layer.initGrads();
        }
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
