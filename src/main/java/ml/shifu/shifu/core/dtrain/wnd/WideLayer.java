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
public class WideLayer implements Layer<List<SparseInput>, List<float[]>, List<float[]>, List<float[]>> {

    private List<WideFieldLayer> layers;

    @Override
    public int getOutDim() {
        int len = 0;
        for(WideFieldLayer layer: layers) {
            len += layer.getOutDim();
        }
        return len;
    }

    @Override
    public List<float[]> forward(List<SparseInput> inputList) {
        assert this.layers.size() == inputList.size();
        List<float[]> list = new ArrayList<float[]>();
        for(int i = 0; i < layers.size(); i++) {
            list.add(this.layers.get(i).forward(inputList.get(i)));
        }
        return list;
    }

    @Override
    public List<float[]> backward(List<float[]> backInputList) {
        assert this.layers.size() == backInputList.size();
        List<float[]> list = new ArrayList<float[]>();
        for(int i = 0; i < layers.size(); i++) {
            list.add(this.layers.get(i).backward(backInputList.get(i)));
        }
        return list;
    }

}
