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

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * TODO
 * 
 * @author pengzhang
 * 
 */
public class WideAndDeep {

    private DenseInputLayer dil;

    @SuppressWarnings("rawtypes")
    private List<Layer> hiddenLayers;

    private DenseLayer finalLayer;

    private EmbedCombinedLayer ecl;

    private WideLayer wl;

    // TODO support wide-only and dnn-only case
    public WideAndDeep(List<ColumnConfig> columnConfigList, int numericalSize, List<Integer> embedColumnIds,
            List<Integer> embedOutputs, List<Integer> wideColumnIds, List<Integer> hiddenNodes,
            List<String> actiFuncs) {
        this.dil = new DenseInputLayer(numericalSize);

        assert embedColumnIds.size() == embedOutputs.size();
        List<EmbedLayer> embedLayers = new ArrayList<EmbedLayer>();
        for(int i = 0; i < embedColumnIds.size(); i++) {
            Integer columnId = embedColumnIds.get(i);
            ColumnConfig config = columnConfigList.get(columnId);
            // +1 to append missing category
            EmbedLayer el = new EmbedLayer(columnId, embedOutputs.get(i), config.getBinCategory().size() + 1);
            embedLayers.add(el);
        }
        this.ecl = new EmbedCombinedLayer(embedLayers);

        List<WideFieldLayer> wfLayers = new ArrayList<WideFieldLayer>();
        for(int i = 0; i < wideColumnIds.size(); i++) {
            Integer columnId = wideColumnIds.get(i);
            ColumnConfig config = columnConfigList.get(columnId);
            WideFieldLayer wfl = new WideFieldLayer(columnId, config.getBinCategory().size() + 1);
            wfLayers.add(wfl);
        }

        this.wl = new WideLayer(wfLayers, new BiasLayer());

        int preHiddenInputs = dil.getOutDim() + ecl.getOutDim();

        assert hiddenNodes.size() == actiFuncs.size();
        for(int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs);
            hiddenLayers.add(denseLayer);
            String acti = actiFuncs.get(i);

            // TODO , add more else;
            if("relu".equalsIgnoreCase(acti)) {
                hiddenLayers.add(new ReLU());
            } else if("sigmoid".equalsIgnoreCase(acti)) {
                hiddenLayers.add(new Sigmoid());
            }
            preHiddenInputs = hiddenOutputs;
        }

        this.finalLayer = new DenseLayer(1, preHiddenInputs);
    }

    @SuppressWarnings("rawtypes")
    public float[] forward(float[] denseInputs, List<SparseInput> embedInputs, List<SparseInput> wideInputs) {
        float[] wlLogits = this.wl.forward(wideInputs);
        float[] dilOuts = this.dil.forward(denseInputs);
        List<float[]> eclOutList = this.ecl.forward(embedInputs);
        float[] inputs = mergeToDenseInputs(dilOuts, eclOutList);
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(i);
            if(layer instanceof DenseLayer) {
                DenseLayer dl = (DenseLayer) layer;
                inputs = dl.forward(inputs);
            } else if(layer instanceof Activiation) {
                Activiation acti = (Activiation) layer;
                inputs = acti.forward(inputs);
            }
        }
        float[] dnnLogits = this.finalLayer.forward(inputs);

        assert wlLogits.length == dnnLogits.length;
        float[] logits = new float[wlLogits.length];
        for(int i = 0; i < logits.length; i++) {
            logits[i] += wlLogits[i] + dnnLogits[i];
        }
        return logits;
    }

    @SuppressWarnings("rawtypes")
    public float[] backward(float[] error) {
        this.wl.backward(error);

        float[] backInputs = this.finalLayer.backward(error);
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1 - i);
            if(layer instanceof DenseLayer) {
                DenseLayer dl = (DenseLayer) layer;
                backInputs = dl.backward(backInputs);
            } else if(layer instanceof Activiation) {
                Activiation acti = (Activiation) layer;
                backInputs = acti.backward(backInputs);
            }
        }
        
        List<float[]> backInputList = splitArray(backInputs);
        this.ecl.backward(backInputList);
        return null;
    }

    /**
     * @param backInputs
     * @return
     */
    private List<float[]> splitArray(float[] backInputs) {
        // TODO Auto-generated method stub
        return null;
    }

    private float[] mergeToDenseInputs(float[] dilOuts, List<float[]> eclOutList) {
        // TODO Auto-generated method stub
        return null;
    }

    /**
     * @return the dil
     */
    public DenseInputLayer getDil() {
        return dil;
    }

    /**
     * @param dil
     *            the dil to set
     */
    public void setDil(DenseInputLayer dil) {
        this.dil = dil;
    }

    /**
     * @return the hiddenLayers
     */
    @SuppressWarnings("rawtypes")
    public List<Layer> getHiddenLayers() {
        return hiddenLayers;
    }

    /**
     * @param hiddenLayers
     *            the hiddenLayers to set
     */
    @SuppressWarnings("rawtypes")
    public void setHiddenLayers(List<Layer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    /**
     * @return the finalLayer
     */
    public DenseLayer getFinalLayer() {
        return finalLayer;
    }

    /**
     * @param finalLayer
     *            the finalLayer to set
     */
    public void setFinalLayer(DenseLayer finalLayer) {
        this.finalLayer = finalLayer;
    }

    /**
     * @return the ecl
     */
    public EmbedCombinedLayer getEcl() {
        return ecl;
    }

    /**
     * @param ecl
     *            the ecl to set
     */
    public void setEcl(EmbedCombinedLayer ecl) {
        this.ecl = ecl;
    }

    /**
     * @return the wl
     */
    public WideLayer getWl() {
        return wl;
    }

    /**
     * @param wl
     *            the wl to set
     */
    public void setWl(WideLayer wl) {
        this.wl = wl;
    }

}
