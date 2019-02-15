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
 * {@link WideAndDeep} graph definition which is for whole network including deep side and wide side.
 * 
 * <p>
 * WideAndDeep is split into dense inputs, embed inputs and wide inputs. With dense inputs + embed inputs, DNN is
 * constructed according to hidden layer settings. Wide inputs are for wide part computations as LR (no hidden layer).
 * 
 * <p>
 * TODO general chart
 * TODO how gradients and computation logic
 * TODO how to scale?
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WideAndDeep implements WeightInitializable {

    private DenseInputLayer dil;

    @SuppressWarnings("rawtypes")
    private List<Layer> hiddenLayers;

    private DenseLayer finalLayer;

    private EmbedLayer ecl;

    private WideLayer wl;

    private List<ColumnConfig> columnConfigList;

    private int numericalSize;

    private List<Integer> denseColumnIds;

    private List<Integer> embedColumnIds;

    private List<Integer> embedOutputs;

    private List<Integer> wideColumnIds;

    private List<Integer> hiddenNodes;

    private List<String> actiFuncs;

    private float l2reg;

    public WideAndDeep(WideAndDeep wnd) {
        // TODO
    }

    public WideAndDeep() {
    }

    @SuppressWarnings("rawtypes")
    public WideAndDeep(List<Layer> hiddenLayers, DenseLayer finalLayer, EmbedLayer ecl, WideLayer wl,
            List<ColumnConfig> columnConfigList, int numericalSize, List<Integer> denseColumnIds,
            List<Integer> embedColumnIds, List<Integer> embedOutputs, List<Integer> wideColumnIds,
            List<Integer> hiddenNodes, List<String> actiFuncs, float l2reg) {
        this.hiddenLayers = hiddenLayers;
        this.finalLayer = finalLayer;
        this.ecl = ecl;
        this.wl = wl;
        this.columnConfigList = columnConfigList;
        this.numericalSize = numericalSize;
        this.denseColumnIds = denseColumnIds;
        this.embedColumnIds = embedColumnIds;
        this.embedOutputs = embedOutputs;
        this.wideColumnIds = wideColumnIds;
        this.hiddenNodes = hiddenNodes;
        this.actiFuncs = actiFuncs;
        this.l2reg = l2reg;

        assert embedColumnIds.size() == embedOutputs.size();
        assert hiddenNodes.size() == actiFuncs.size();
    }

    // TODO support wide-only and dnn-only case
    public WideAndDeep(List<ColumnConfig> columnConfigList, int numericalSize, List<Integer> denseColumnIds,
            List<Integer> embedColumnIds, List<Integer> embedOutputs, List<Integer> wideColumnIds,
            List<Integer> hiddenNodes, List<String> actiFuncs, float l2reg) {
        this.columnConfigList = columnConfigList;
        this.numericalSize = numericalSize;
        this.denseColumnIds = denseColumnIds;
        this.embedColumnIds = embedColumnIds;
        this.embedOutputs = embedOutputs;
        this.wideColumnIds = wideColumnIds;
        this.hiddenNodes = hiddenNodes;
        this.actiFuncs = actiFuncs;
        this.l2reg = l2reg;

        this.dil = new DenseInputLayer(numericalSize);

        assert embedColumnIds.size() == embedOutputs.size();
        List<EmbedFieldLayer> embedLayers = new ArrayList<>();
        for(int i = 0; i < embedColumnIds.size(); i++) {
            Integer columnId = embedColumnIds.get(i);
            ColumnConfig config = columnConfigList.get(columnId);
            // +1 to append missing category
            EmbedFieldLayer el = new EmbedFieldLayer(columnId, embedOutputs.get(i), config.getBinCategory().size() + 1);
            embedLayers.add(el);
        }
        this.ecl = new EmbedLayer(embedLayers);

        List<WideFieldLayer> wfLayers = new ArrayList<>();
        for(Integer columnId: wideColumnIds) {
            ColumnConfig config = columnConfigList.get(columnId);
            WideFieldLayer wfl = new WideFieldLayer(columnId, config.getBinCategory().size() + 1, l2reg);
            wfLayers.add(wfl);
        }

        this.wl = new WideLayer(wfLayers, new BiasLayer());

        int preHiddenInputs = dil.getOutDim() + ecl.getOutDim();

        assert hiddenNodes.size() == actiFuncs.size();
        this.hiddenLayers = new ArrayList<>(hiddenNodes.size() * 2);
        for(int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs, l2reg);
            this.hiddenLayers.add(denseLayer);
            String acti = actiFuncs.get(i);

            // TODO add more else
            if("relu".equalsIgnoreCase(acti)) {
                this.hiddenLayers.add(new ReLU());
            } else if("sigmoid".equalsIgnoreCase(acti)) {
                this.hiddenLayers.add(new Sigmoid());
            }
            preHiddenInputs = hiddenOutputs;
        }

        this.finalLayer = new DenseLayer(1, preHiddenInputs, l2reg);
    }

    @SuppressWarnings("rawtypes")
    public float[] forward(float[] denseInputs, List<SparseInput> embedInputs, List<SparseInput> wideInputs) {
        // wide layer forward
        float[] wlLogits = this.wl.forward(wideInputs);

        // deep layer forward
        float[] dilOuts = this.dil.forward(denseInputs);
        List<float[]> eclOutList = this.ecl.forward(embedInputs);
        float[] inputs = mergeToDenseInputs(dilOuts, eclOutList);
        for(Layer layer: this.hiddenLayers) {
            if(layer instanceof DenseLayer) {
                DenseLayer dl = (DenseLayer) layer;
                inputs = dl.forward(inputs);
            } else if(layer instanceof Activation) {
                Activation acti = (Activation) layer;
                inputs = acti.forward(inputs);
            }
        }
        float[] dnnLogits = this.finalLayer.forward(inputs);

        // merge wide and deep together
        assert wlLogits.length == dnnLogits.length;
        float[] logits = new float[wlLogits.length];
        for(int i = 0; i < logits.length; i++) {
            logits[i] += wlLogits[i] + dnnLogits[i];
        }
        return logits;
    }

    @SuppressWarnings("rawtypes")
    public float[] backward(float[] error, float sig) {
        // wide layer backward, as wide layer in LR actually in backward, only gradients computation is needed.
        this.wl.backward(error, sig);

        // deep layer backward, for gradients computation inside of each layer
        float[] backInputs = this.finalLayer.backward(error, sig);
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1 - i);
            if(layer instanceof DenseLayer) {
                DenseLayer dl = (DenseLayer) layer;
                backInputs = dl.backward(backInputs, sig);
            } else if(layer instanceof Activation) {
                Activation acti = (Activation) layer;
                backInputs = acti.backward(backInputs, sig);
            }
        }

        // embedding layer backward, gradients computation
        List<float[]> backInputList = splitArray(this.dil.getOutDim(), this.ecl.getEmbedLayers(), backInputs);
        this.ecl.backward(backInputList, sig);

        // no need return final backward outputs as gradients are computed well
        return null;
    }

    /**
     * Initialize gradients for training of each epoch
     */
    @SuppressWarnings("rawtypes")
    public void initGrads() {
        for(Layer layer: hiddenLayers) {
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initGrads();
            }
        }
        this.finalLayer.initGrads();
        this.ecl.initGrads();
        this.wl.initGrads();
    }

    private List<float[]> splitArray(int outDim, List<EmbedFieldLayer> embedLayers, float[] backInputs) {
        List<float[]> results = new ArrayList<>();
        int srcPos = outDim;
        for(EmbedFieldLayer el: embedLayers) {
            float[] elBackInputs = new float[el.getIn()];
            System.arraycopy(backInputs, srcPos, elBackInputs, 0, elBackInputs.length);
            srcPos += elBackInputs.length;
            results.add(elBackInputs);
        }
        return results;
    }

    private float[] mergeToDenseInputs(float[] dilOuts, List<float[]> eclOutList) {
        int len = dilOuts.length;
        for(float[] fs: eclOutList) {
            len += fs.length;
        }

        float[] results = new float[len];

        // copy dense
        System.arraycopy(dilOuts, 0, results, 0, dilOuts.length);

        // copy embed
        int currIndex = dilOuts.length;
        for(float[] fs: eclOutList) {
            System.arraycopy(fs, 0, results, currIndex, fs.length);
            currIndex += fs.length;
        }
        return results;
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
    public EmbedLayer getEcl() {
        return ecl;
    }

    /**
     * @param ecl
     *            the ecl to set
     */
    public void setEcl(EmbedLayer ecl) {
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

    /**
     * @return the columnConfigList
     */
    public List<ColumnConfig> getColumnConfigList() {
        return columnConfigList;
    }

    /**
     * @param columnConfigList
     *            the columnConfigList to set
     */
    public void setColumnConfigList(List<ColumnConfig> columnConfigList) {
        this.columnConfigList = columnConfigList;
    }

    /**
     * @return the numericalSize
     */
    public int getNumericalSize() {
        return numericalSize;
    }

    /**
     * @param numericalSize
     *            the numericalSize to set
     */
    public void setNumericalSize(int numericalSize) {
        this.numericalSize = numericalSize;
    }

    public List<Integer> getDenseColumnIds() {
        return denseColumnIds;
    }

    /**
     * @param denseColumnIds
     *            the denseColumnIds to set
     */
    public void setDenseColumnIds(List<Integer> denseColumnIds) {
        this.denseColumnIds = denseColumnIds;
    }

    /**
     * @return the embedColumnIds
     */
    public List<Integer> getEmbedColumnIds() {
        return embedColumnIds;
    }

    /**
     * @param embedColumnIds
     *            the embedColumnIds to set
     */
    public void setEmbedColumnIds(List<Integer> embedColumnIds) {
        this.embedColumnIds = embedColumnIds;
    }

    /**
     * @return the embedOutputs
     */
    public List<Integer> getEmbedOutputs() {
        return embedOutputs;
    }

    /**
     * @param embedOutputs
     *            the embedOutputs to set
     */
    public void setEmbedOutputs(List<Integer> embedOutputs) {
        this.embedOutputs = embedOutputs;
    }

    /**
     * @return the wideColumnIds
     */
    public List<Integer> getWideColumnIds() {
        return wideColumnIds;
    }

    /**
     * @param wideColumnIds
     *            the wideColumnIds to set
     */
    public void setWideColumnIds(List<Integer> wideColumnIds) {
        this.wideColumnIds = wideColumnIds;
    }

    /**
     * @return the hiddenNodes
     */
    public List<Integer> getHiddenNodes() {
        return hiddenNodes;
    }

    /**
     * @param hiddenNodes
     *            the hiddenNodes to set
     */
    public void setHiddenNodes(List<Integer> hiddenNodes) {
        this.hiddenNodes = hiddenNodes;
    }

    /**
     * @return the actiFuncs
     */
    public List<String> getActiFuncs() {
        return actiFuncs;
    }

    /**
     * @param actiFuncs
     *            the actiFuncs to set
     */
    public void setActiFuncs(List<String> actiFuncs) {
        this.actiFuncs = actiFuncs;
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

    public void updateWeights(WideAndDeep wnd) {
        // TODO copy weights from wnd object and set it in current wide and deep, update weights from master
    }

    public void updateWeights(WNDParams params) {
        updateWeights(params.getWnd());
        // after update weights, gradients should be re newed.
        this.initGrads();
    }

    /**
     * TODO: init the weights in WideAndDeeep Model and it's sub module
     */
    public void initWeights() {
        // TODO
        String defaultMode = "get_from_configuration";
        initWeight(defaultMode);
    }

    @SuppressWarnings("rawtypes")
    @Override
    public void initWeight(String policy) {
        for(Layer layer: this.hiddenLayers) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight(policy);
            }
        }
        this.finalLayer.initWeight(policy);
        this.ecl.initWeight(policy);
        this.wl.initWeight(policy);
    }
}
