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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.wnd.activation.Activation;
import ml.shifu.shifu.core.dtrain.wnd.activation.ReLU;
import ml.shifu.shifu.core.dtrain.wnd.activation.Sigmoid;
import ml.shifu.shifu.core.dtrain.wnd.optimization.Optimizer;
import ml.shifu.shifu.util.Tuple;

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
public class WideAndDeep implements WeightInitializer, Bytable, Combinable<WideAndDeep> {

    private DenseInputLayer dil;

    @SuppressWarnings("rawtypes")
    private List<Layer> hiddenLayers;

    private DenseLayer finalLayer;

    private EmbedLayer ecl;

    private WideLayer wl;

    private Map<Integer, Integer> idBinCateSizeMap;

    private int numericalSize;

    private List<Integer> denseColumnIds;

    private List<Integer> embedColumnIds;

    private List<Integer> embedOutputs;

    private List<Integer> wideColumnIds;

    private List<Integer> hiddenNodes;

    private List<String> actiFuncs;

    private float l2reg;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    public WideAndDeep(WideAndDeep wnd) {
        // TODO
    }

    public WideAndDeep() {
    }

    @SuppressWarnings("rawtypes")
    public WideAndDeep(List<Layer> hiddenLayers, DenseLayer finalLayer, EmbedLayer ecl, WideLayer wl,
            Map<Integer, Integer> idBinCateSizeMap, int numericalSize, List<Integer> denseColumnIds,
            List<Integer> embedColumnIds, List<Integer> embedOutputs, List<Integer> wideColumnIds,
            List<Integer> hiddenNodes, List<String> actiFuncs, float l2reg) {
        this.hiddenLayers = hiddenLayers;
        this.finalLayer = finalLayer;
        this.ecl = ecl;
        this.wl = wl;
        this.idBinCateSizeMap = idBinCateSizeMap;
        this.numericalSize = numericalSize;
        this.denseColumnIds = denseColumnIds;
        this.embedColumnIds = embedColumnIds;
        this.embedOutputs = embedOutputs;
        this.wideColumnIds = wideColumnIds;
        this.hiddenNodes = hiddenNodes;
        this.actiFuncs = actiFuncs;
        this.l2reg = l2reg;

        AssertUtils.assertListNotNullAndSizeEqual(embedColumnIds, embedOutputs);
        AssertUtils.assertListNotNullAndSizeEqual(hiddenLayers, actiFuncs);
    }

    // TODO support wide-only and dnn-only case
    public WideAndDeep(Map<Integer, Integer> idBinCateSizeMap, int numericalSize, List<Integer> denseColumnIds,
            List<Integer> embedColumnIds, List<Integer> embedOutputs, List<Integer> wideColumnIds,
            List<Integer> hiddenNodes, List<String> actiFuncs, float l2reg) {
        this.idBinCateSizeMap = idBinCateSizeMap;
        this.numericalSize = numericalSize;
        this.denseColumnIds = denseColumnIds;
        this.embedColumnIds = embedColumnIds;
        this.embedOutputs = embedOutputs;
        this.wideColumnIds = wideColumnIds;
        this.hiddenNodes = hiddenNodes;
        this.actiFuncs = actiFuncs;
        this.l2reg = l2reg;

        this.dil = new DenseInputLayer(numericalSize);

        AssertUtils.assertListNotNullAndSizeEqual(embedColumnIds, embedOutputs);
        List<EmbedFieldLayer> embedLayers = new ArrayList<>();
        for(int i = 0; i < embedColumnIds.size(); i++) {
            Integer columnId = embedColumnIds.get(i);
            // +1 to append missing category
            EmbedFieldLayer el = new EmbedFieldLayer(columnId, embedOutputs.get(i),
                    this.idBinCateSizeMap.get(columnId) + 1);
            embedLayers.add(el);
        }
        this.ecl = new EmbedLayer(embedLayers);

        List<WideFieldLayer> wfLayers = new ArrayList<>();
        for(Integer columnId: wideColumnIds) {
            WideFieldLayer wfl = new WideFieldLayer(columnId, this.idBinCateSizeMap.get(columnId) + 1, l2reg);
            wfLayers.add(wfl);
        }

        WideDenseLayer wdl = new WideDenseLayer(this.denseColumnIds, this.denseColumnIds.size(), l2reg);
        this.wl = new WideLayer(wfLayers, wdl, new BiasLayer());

        int preHiddenInputs = dil.getOutDim() + ecl.getOutDim();

        AssertUtils.assertListNotNullAndSizeEqual(hiddenNodes, actiFuncs);
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

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public float[] forward(float[] denseInputs, List<SparseInput> embedInputs, List<SparseInput> wideInputs) {
        // wide layer forward
        float[] wlLogits = this.wl.forward(new Tuple(wideInputs, denseInputs));

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
        AssertUtils.assertFloatArrayNotNullAndLengthEqual(wlLogits, dnnLogits);
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
     * @return the idBinCateSizeMap
     */
    public Map<Integer, Integer> getIdBinCateSizeMap() {
        return idBinCateSizeMap;
    }

    /**
     * @param idBinCateSizeMap
     *            the idBinCateSizeMap to set
     */
    public void setIdBinCateSizeMap(Map<Integer, Integer> idBinCateSizeMap) {
        this.idBinCateSizeMap = idBinCateSizeMap;
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

    /**
     * @return the serializationType
     */
    public SerializationType getSerializationType() {
        return serializationType;
    }

    /**
     * @param serializationType
     *            the serializationType to set
     */
    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
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
        InitMethod defaultMode = InitMethod.ZERO_ONE_RANGE_RANDOM;
        initWeight(defaultMode);
    }

    @SuppressWarnings("rawtypes")
    @Override
    public void initWeight(InitMethod method) {
        for(Layer layer: this.hiddenLayers) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight(method);
            }
        }
        this.finalLayer.initWeight(method);
        this.ecl.initWeight(method);
        this.wl.initWeight(method);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @SuppressWarnings("rawtypes")
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.serializationType.getValue());
        writeLayerWithNuLLCheck(out, dil);

        if(actiFuncs == null) {
            out.writeInt(0);
        } else {
            out.writeInt(actiFuncs.size());
            for(String acti: actiFuncs) {
                out.writeUTF(acti);
            }
        }
        if(this.hiddenLayers == null) {
            out.writeInt(0);
        } else {
            out.writeInt(hiddenLayers.size());
            for(Layer layer: hiddenLayers) {
                if(layer instanceof DenseLayer) { // Activation is constructed from actiFuncs
                    DenseLayer denseLayer = (DenseLayer) layer;
                    denseLayer.write(out, serializationType);
                }
            }
        }

        writeLayerWithNuLLCheck(out, finalLayer);
        writeLayerWithNuLLCheck(out, ecl);
        writeLayerWithNuLLCheck(out, wl);

        if(this.serializationType == SerializationType.MODEL_SPEC) {
            if(idBinCateSizeMap == null) {
                out.writeInt(0);
            } else {
                out.writeInt(idBinCateSizeMap.size());
                for(Entry<Integer, Integer> entry: idBinCateSizeMap.entrySet()) {
                    out.writeInt(entry.getKey());
                    out.writeInt(entry.getValue());
                }
            }
            out.writeInt(numericalSize);
            SerializationUtil.writeIntList(out, denseColumnIds);
            SerializationUtil.writeIntList(out, embedColumnIds);
            SerializationUtil.writeIntList(out, embedOutputs);
            SerializationUtil.writeIntList(out, wideColumnIds);
            SerializationUtil.writeIntList(out, hiddenNodes);
            out.writeFloat(l2reg);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @SuppressWarnings("rawtypes")
    @Override
    public void readFields(DataInput in) throws IOException {
        this.serializationType = SerializationType.getSerializationType(in.readInt());

        if(this.dil == null) {
            this.dil = new DenseInputLayer();
        }
        this.dil.readFields(in, serializationType);

        if(actiFuncs == null) {
            actiFuncs = new ArrayList<String>();
        }
        actiFuncs.clear();
        for(int i = 0; i < in.readInt(); i++) {
            actiFuncs.add(in.readUTF());
        }

        List<DenseLayer> hiddenDenseLayers = new ArrayList<DenseLayer>();
        if(hiddenLayers != null) {
            // get existing dense hidden layers to save memory
            for(Layer layer: hiddenLayers) {
                if(layer instanceof DenseLayer) {
                    hiddenDenseLayers.add((DenseLayer) layer);
                }
            }
            hiddenLayers.clear();
        } else {
            hiddenLayers = new ArrayList<Layer>();
        }
        for(int i = 0; i < in.readInt(); i++) {
            if(hiddenDenseLayers.size() > i) {
                hiddenDenseLayers.get(i).readFields(in, serializationType);
            } else {
                DenseLayer tmpLayer = new DenseLayer();
                tmpLayer.readFields(in, serializationType);
                hiddenDenseLayers.add(tmpLayer);
            }
        }
        hiddenLayers = new ArrayList<Layer>();
        for(int i = 0; i < hiddenDenseLayers.size(); i++) {
            hiddenLayers.add(hiddenDenseLayers.get(i));
            String acti = actiFuncs.get(i);

            if("relu".equalsIgnoreCase(acti)) {
                this.hiddenLayers.add(new ReLU());
            } else if("sigmoid".equalsIgnoreCase(acti)) {
                this.hiddenLayers.add(new Sigmoid());
            }
        }

        finalLayer = (DenseLayer) readLayerWithNullCheck(in, finalLayer == null ? new DenseLayer() : finalLayer);
        ecl = (EmbedLayer) readLayerWithNullCheck(in, ecl == null ? new EmbedLayer() : ecl);
        wl = (WideLayer) readLayerWithNullCheck(in, wl == null ? new WideLayer() : wl);

        if(serializationType == SerializationType.MODEL_SPEC) {
            if(idBinCateSizeMap == null) {
                idBinCateSizeMap = new HashMap<Integer, Integer>();
            }
            idBinCateSizeMap.clear();
            for(int i = 0; i < in.readInt(); i++) {
                idBinCateSizeMap.put(in.readInt(), in.readInt());
            }
            numericalSize = in.readInt();
            denseColumnIds = SerializationUtil.readIntList(in, denseColumnIds);
            embedColumnIds = SerializationUtil.readIntList(in, embedColumnIds);
            embedOutputs = SerializationUtil.readIntList(in, embedOutputs);
            wideColumnIds = SerializationUtil.readIntList(in, wideColumnIds);
            hiddenNodes = SerializationUtil.readIntList(in, hiddenNodes);
            l2reg = in.readFloat();
        }
    }

    /**
     * Write layer with null check.
     */
    @SuppressWarnings("rawtypes")
    private void writeLayerWithNuLLCheck(DataOutput out, AbstractLayer layer) throws IOException {
        if(layer == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            layer.write(out, this.serializationType);
        }
    }

    /**
     * Read layer with null check.
     * 
     * @param in
     * @param layer
     *            the layer to hold serialized data. This value should not be null.
     * @return de-serialized layer instance
     * @throws IOException
     */
    @SuppressWarnings("rawtypes")
    private AbstractLayer readLayerWithNullCheck(DataInput in, AbstractLayer layer) throws IOException {
        if(in.readBoolean()) {
            layer.readFields(in, serializationType);
        }
        return layer;
    }

    @SuppressWarnings("rawtypes")
    @Override
    public WideAndDeep combine(WideAndDeep from) {
        this.dil = this.dil.combine(from.getDil());

        List<Layer> fhl = from.getHiddenLayers();
        int hlSize = hiddenLayers.size();
        for(int i = 0; i < hlSize; i++) {
            if(hiddenLayers.get(i) instanceof DenseLayer) {
                Layer nLayer = ((DenseLayer) hiddenLayers.get(i)).combine((DenseLayer) fhl.get(i));
                hiddenLayers.add(i, nLayer);
            }
        }

        this.finalLayer = this.finalLayer.combine(from.getFinalLayer());
        this.ecl = this.ecl.combine(from.getEcl());
        this.wl = this.wl.combine(from.getWl());
        return this;
    }

    @SuppressWarnings("rawtypes")
    public void update(WideAndDeep gradWnd, Optimizer optimizer) {
        this.dil.update(gradWnd.getDil(), optimizer);

        List<Layer> gradHLs = gradWnd.getHiddenLayers();
        int hlSize = hiddenLayers.size();
        for(int i = 0; i < hlSize; i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if(tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).update((DenseLayer) gradHLs.get(i), optimizer);
            }
        }

        this.finalLayer.update(gradWnd.getFinalLayer(), optimizer);
        this.ecl.update(gradWnd.getEcl(), optimizer);
        this.wl.update(gradWnd.getWl(), optimizer);
    }

}
