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

import java.util.ArrayList;
import java.util.List;

/**
 * TODO
 * 
 * @author pengzhang
 */
public class WideLayer implements Layer<List<SparseInput>, float[], float[], List<float[]>> {

    private List<WideFieldLayer> layers;

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
        assert this.getLayers().size() == inputList.size();
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
    public List<float[]> backward(float[] backInputs) {
        // below backward call is for gradients computation in WideFieldLayer and BiasLayer
        List<float[]> list = new ArrayList<float[]>();
        for(int i = 0; i < getLayers().size(); i++) {
            list.add(this.getLayers().get(i).backward(backInputs));
        }
        list.add(new float[] { bias.backward(backInputs[0]) });
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

}
