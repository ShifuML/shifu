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
 * {@link EmbedLayer} merges all embedding layers together and distributes forward and backward computation to
 * {@link EmbedFieldLayer}s.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class EmbedLayer
        implements Layer<List<SparseInput>, List<float[]>, List<float[]>, List<float[]>>, WeightInitialisable, Bytable {

    /**
     * List of embed layer which is for each embed column.
     */
    private List<EmbedFieldLayer> embedLayers;

    public EmbedLayer(List<EmbedFieldLayer> embedLayers) {
        this.embedLayers = embedLayers;
    }

    @Override
    public int getOutDim() {
        int len = 0;
        for(EmbedFieldLayer embedLayer: getEmbedLayers()) {
            len += embedLayer.getOutDim();
        }
        return len;
    }

    @Override
    public List<float[]> forward(List<SparseInput> inputList) {
        AssertUtils.assertListNotNullAndSizeEqual(this.getEmbedLayers(), inputList);
        List<float[]> list = new ArrayList<>();
        for(int i = 0; i < this.getEmbedLayers().size(); i++) {
            list.add(this.getEmbedLayers().get(i).forward(inputList.get(i)));
        }
        return list;
    }

    @Override
    public List<float[]> backward(List<float[]> backInputList, float sig) {
        AssertUtils.assertListNotNullAndSizeEqual(this.getEmbedLayers(), backInputList);
        List<float[]> list = new ArrayList<>();
        for(int i = 0; i < this.getEmbedLayers().size(); i++) {
            list.add(this.getEmbedLayers().get(i).backward(backInputList.get(i), sig));
        }
        return list;
    }

    /**
     * @return the embedLayers
     */
    public List<EmbedFieldLayer> getEmbedLayers() {
        return embedLayers;
    }

    /**
     * @param embedLayers
     *            the embedLayers to set
     */
    public void setEmbedLayers(List<EmbedFieldLayer> embedLayers) {
        this.embedLayers = embedLayers;
    }

    @Override public void initWeight(InitMethod method) {
        for(EmbedFieldLayer embedFieldLayer: this.embedLayers) {
            embedFieldLayer.initWeight(method);
        }
    }

    public void initGrads() {
        for(EmbedFieldLayer embedFieldLayer: this.embedLayers) {
            embedFieldLayer.initGrads();
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
