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
public class EmbedCombinedLayer implements Layer<List<SparseInput>, List<float[]>, List<float[]>, List<float[]>> {

    private List<EmbedLayer> embedLayers;

    public EmbedCombinedLayer(List<EmbedLayer> embedLayers) {
        this.setEmbedLayers(embedLayers);
    }

    @Override
    public int getOutDim() {
        int len = 0;
        for(EmbedLayer embedLayer: getEmbedLayers()) {
            len += embedLayer.getOutDim();
        }
        return len;
    }

    @Override
    public List<float[]> forward(List<SparseInput> inputList) {
        assert this.getEmbedLayers().size() == inputList.size();
        List<float[]> list = new ArrayList<float[]>();
        for(int i = 0; i < this.getEmbedLayers().size(); i++) {
            list.add(this.getEmbedLayers().get(i).forward(inputList.get(i)));
        }
        return list;
    }

    @Override
    public List<float[]> backward(List<float[]> backInputList) {
        assert this.getEmbedLayers().size() == backInputList.size();
        List<float[]> list = new ArrayList<float[]>();
        for(int i = 0; i < this.getEmbedLayers().size(); i++) {
            list.add(this.getEmbedLayers().get(i).backward(backInputList.get(i)));
        }
        return list;
    }

    /**
     * @return the embedLayers
     */
    public List<EmbedLayer> getEmbedLayers() {
        return embedLayers;
    }

    /**
     * @param embedLayers the embedLayers to set
     */
    public void setEmbedLayers(List<EmbedLayer> embedLayers) {
        this.embedLayers = embedLayers;
    }

}
