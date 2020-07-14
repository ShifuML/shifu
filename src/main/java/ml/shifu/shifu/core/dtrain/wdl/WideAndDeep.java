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

import static ml.shifu.shifu.core.dtrain.layer.SerializationUtil.NULL;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.layer.AbstractLayer;
import ml.shifu.shifu.core.dtrain.layer.BiasLayer;
import ml.shifu.shifu.core.dtrain.layer.DenseInputLayer;
import ml.shifu.shifu.core.dtrain.layer.DenseLayer;
import ml.shifu.shifu.core.dtrain.layer.EmbedFieldLayer;
import ml.shifu.shifu.core.dtrain.layer.EmbedLayer;
import ml.shifu.shifu.core.dtrain.layer.Layer;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;
import ml.shifu.shifu.core.dtrain.layer.SerializationUtil;
import ml.shifu.shifu.core.dtrain.layer.SparseInput;
import ml.shifu.shifu.core.dtrain.layer.WeightInitializer;
import ml.shifu.shifu.core.dtrain.layer.WideDenseLayer;
import ml.shifu.shifu.core.dtrain.layer.WideFieldLayer;
import ml.shifu.shifu.core.dtrain.layer.WideLayer;
import ml.shifu.shifu.core.dtrain.layer.activation.Activation;
import ml.shifu.shifu.core.dtrain.layer.activation.ActivationFactory;
import ml.shifu.shifu.core.dtrain.layer.optimization.Optimizer;
import ml.shifu.shifu.core.dtrain.layer.optimization.PropOptimizer;
import ml.shifu.shifu.core.dtrain.loss.LossType;
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
public class WideAndDeep
        implements WeightInitializer<WideAndDeep>, Bytable, Combinable<WideAndDeep>, PropOptimizer<WideAndDeep> {

    private static final Logger LOG = LoggerFactory.getLogger(WideAndDeep.class);

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

    private double l2reg;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    private int index = 0;

    /**
     * A new layer to combine both wide score and deep score with different weights if enabling both wide and deep parts
     */
    private DenseLayer wdLayer;

    /**
     * If enable wide layer, if wideEnable=true but deepEnable=false, such WideAnDeeo is only for wide layer (LR).
     */
    boolean wideEnable = true;

    /**
     * If enable deep layer, if deepEnable=true but wideEnable=false, like Shifu NN.
     */
    boolean deepEnable = true;

    /**
     * Only works when deepEnable=true, if embedEnable=false then embed columns would be removed.
     */
    boolean embedEnable = true;

    /**
     * Enable dense fields in WideLayer when {@link #wideDenseEnable} is true, by default is true.
     */
    private boolean wideDenseEnable = true;

    private boolean isDebug = false;

    /**
     * Flat spot value to smooth lr derived function: result * (1 - result): This value sometimes may be close to zero.
     * Add flat sport to improve it: result * (1 - result) + 0.1d
     */
    private static final double FLAT_SPOT_VALUE = 0.1d;

    public WideAndDeep() {
    }

    @SuppressWarnings("rawtypes")
    public WideAndDeep(boolean wideEnable, boolean deepEnable, boolean embedEnable, boolean wideDenseEnable,
            List<Layer> hiddenLayers, DenseLayer finalLayer, EmbedLayer ecl, WideLayer wl,
            Map<Integer, Integer> idBinCateSizeMap, int numericalSize, List<Integer> denseColumnIds,
            List<Integer> embedColumnIds, List<Integer> embedOutputs, List<Integer> wideColumnIds,
            List<Integer> hiddenNodes, List<String> actiFuncs, double l2reg) {
        this.wideEnable = wideEnable;
        this.deepEnable = deepEnable;
        this.embedEnable = embedEnable;
        this.wideDenseEnable = wideDenseEnable;
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

        if(this.wideEnable && this.deepEnable) {
            this.wdLayer = new DenseLayer(1, 2, l2reg);
        }

        AssertUtils.assertListNotNullAndSizeEqual(embedColumnIds, embedOutputs);
        AssertUtils.assertListNotNullAndSizeEqual(hiddenLayers, actiFuncs);
    }

    public WideAndDeep(boolean wideEnable, boolean deepEnable, boolean embedEnable, boolean wideDenseEnable,
            Map<Integer, Integer> idBinCateSizeMap, int numericalSize, List<Integer> denseColumnIds,
            List<Integer> embedColumnIds, List<Integer> embedOutputs, List<Integer> wideColumnIds,
            List<Integer> hiddenNodes, List<String> actiFuncs, double l2reg) {
        this.wideEnable = wideEnable;
        this.deepEnable = deepEnable;
        this.embedEnable = embedEnable;
        this.wideDenseEnable = wideDenseEnable;
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
        this.wl = new WideLayer(wfLayers, wdl, new BiasLayer(), this.wideDenseEnable);

        int preHiddenInputs;
        if(this.embedEnable) {
            preHiddenInputs = dil.getOutDim() + ecl.getOutDim();
        } else {
            preHiddenInputs = dil.getOutDim();
        }

        AssertUtils.assertListNotNullAndSizeEqual(hiddenNodes, actiFuncs);
        this.hiddenLayers = new ArrayList<>(hiddenNodes.size() * 2);
        for(int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs, l2reg);
            this.hiddenLayers.add(denseLayer);
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(actiFuncs.get(i)));
            preHiddenInputs = hiddenOutputs;
        }

        this.finalLayer = new DenseLayer(1, preHiddenInputs, l2reg);

        if(this.wideEnable && this.deepEnable) {
            this.wdLayer = new DenseLayer(1, 2, l2reg);
        }
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public double[] forward(double[] denseInputs, List<SparseInput> embedInputs, List<SparseInput> wideInputs) {
        double[] wlLogits = null;
        if(this.wideEnable) {
            wlLogits = this.wl.forward(new Tuple(wideInputs, denseInputs));
        }

        if(!this.deepEnable) { // wide only mode
            return wlLogits;
        } else { // deep only mode or wide and deep mode
            double[] dilOuts = this.dil.forward(denseInputs);
            List<double[]> eclOutList = null;
            double[] inputs = null;
            if(this.embedEnable) {
                eclOutList = this.ecl.forward(embedInputs);
                inputs = mergeToDenseInputs(dilOuts, eclOutList);
            } else { // not include embed part
                inputs = dilOuts;
            }
            for(Layer layer: this.hiddenLayers) {
                if(layer instanceof DenseLayer) {
                    DenseLayer dl = (DenseLayer) layer;
                    inputs = dl.forward(inputs);
                } else if(layer instanceof Activation) {
                    Activation acti = (Activation) layer;
                    inputs = acti.forward(inputs);
                }
            }

            double[] dnnLogits = this.finalLayer.forward(inputs);
            if(!wideEnable) { // deep only
                return dnnLogits;
            } else { // wide and deep
                AssertUtils.assertDoubleArrayNotNullAndLengthEqual(wlLogits, dnnLogits);
                double[] logits = this.wdLayer.forward(new double[] { wlLogits[0], dnnLogits[0] });
                return logits;
            }
        }
    }

    /**
     * Derived function for sigmoid function.
     */
    private double derivedFunction(double result) {
        return result * (1d - result);
    }

    @SuppressWarnings("rawtypes")
    public double[] backward(double[] predicts, double[] actuals, double sig, LossType lossType) {
        double[] grad2Logits = new double[predicts.length];
        for(int i = 0; i < grad2Logits.length; i++) {
            double error = (predicts[i] - actuals[i]);
            switch(lossType) {
                case LOG:
                    grad2Logits[i] = error * sig * -1d;
                    break;
                case SQUARED:
                default:
                    grad2Logits[i] = error * (derivedFunction(predicts[i]) + FLAT_SPOT_VALUE) * sig * -1d;
                    break;
            }
        }

        if(this.wideEnable && this.deepEnable) {
            double[] backInputs = this.wdLayer.backward(grad2Logits);

            this.wl.backward(new double[] { backInputs[0] });

            backInputs = this.finalLayer.backward(new double[] { backInputs[1] });
            for(int i = 0; i < this.hiddenLayers.size(); i++) {
                Layer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1 - i);
                if(layer instanceof DenseLayer) {
                    backInputs = ((DenseLayer) layer).backward(backInputs);
                } else if(layer instanceof Activation) {
                    backInputs = ((Activation) layer).backward(backInputs);
                }
            }

            if(this.embedEnable) { // embedding layer backward, gradients computation
                List<double[]> backInputList = splitArray(this.dil.getOutDim(), this.ecl.getEmbedLayers(), backInputs);
                this.ecl.backward(backInputList);
            }
        } else {
            // wide layer backward, as wide layer in LR actually in backward, only gradients computation is needed.
            if(this.wideEnable) {
                this.wl.backward(grad2Logits);
            }

            if(this.deepEnable) { // deep layer backward, for gradients computation inside of each layer
                double[] backInputs = this.finalLayer.backward(grad2Logits);
                for(int i = 0; i < this.hiddenLayers.size(); i++) {
                    Layer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1 - i);
                    if(layer instanceof DenseLayer) {
                        backInputs = ((DenseLayer) layer).backward(backInputs);
                    } else if(layer instanceof Activation) {
                        backInputs = ((Activation) layer).backward(backInputs);
                    }
                }

                if(this.embedEnable) { // embedding layer backward, gradients computation
                    List<double[]> backInputList = splitArray(this.dil.getOutDim(), this.ecl.getEmbedLayers(),
                            backInputs);
                    this.ecl.backward(backInputList);
                }
            }
        }

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

        if(this.wideEnable && this.deepEnable) {
            this.wdLayer.initGrads();
        }
    }

    private List<double[]> splitArray(int outDim, List<EmbedFieldLayer> embedLayers, double[] backInputs) {
        List<double[]> results = new ArrayList<>();
        int srcPos = outDim;
        for(EmbedFieldLayer el: embedLayers) {
            double[] elBackInputs = new double[el.getOut()];
            System.arraycopy(backInputs, srcPos, elBackInputs, 0, elBackInputs.length);
            srcPos += elBackInputs.length;
            results.add(elBackInputs);
        }
        return results;
    }

    private double[] mergeToDenseInputs(double[] dilOuts, List<double[]> eclOutList) {
        int len = dilOuts.length;
        for(double[] fs: eclOutList) {
            len += fs.length;
        }

        double[] results = new double[len];

        // copy dense
        System.arraycopy(dilOuts, 0, results, 0, dilOuts.length);

        // copy embed
        int currIndex = dilOuts.length;
        for(double[] fs: eclOutList) {
            System.arraycopy(fs, 0, results, currIndex, fs.length);
            currIndex += fs.length;
        }
        return results;
    }

    /**
     * @return the isDebug
     */
    public boolean isDebug() {
        return isDebug;
    }

    /**
     * @param isDebug
     *            the isDebug to set
     */
    public void setDebug(boolean isDebug) {
        this.isDebug = isDebug;
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
    public double getL2reg() {
        return l2reg;
    }

    /**
     * @param l2reg
     *            the l2reg to set
     */
    public void setL2reg(double l2reg) {
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

    public int getInputNum() {
        return this.denseColumnIds.size() + this.wideColumnIds.size();
    }

    public void updateWeights(WideAndDeep wnd) {
        this.initWeight(wnd);
    }

    public void updateWeights(WDLParams params) {
        updateWeights(params.getWnd());
        // after update weights, gradients should be re newed.
        this.initGrads();
    }

    /**
     * Init the weights in WideAndDeep Model and it's sub module
     */
    @SuppressWarnings("rawtypes")
    public void initWeights() {
        InitMethod defaultMode = InitMethod.NEGATIVE_POSITIVE_ONE_RANGE_RANDOM;
        initWeight(defaultMode);

        int hiddenCount = 0;
        for(Layer layer: this.hiddenLayers) {
            if(layer instanceof DenseLayer) {
                hiddenCount += ((DenseLayer) layer).getOut();
            }
        }

        // can't really do much, use regular randomization
        if(hiddenCount < 1) {
            return;
        }

        int inputCount = 0;
        if(this.embedEnable) {
            inputCount = dil.getOutDim() + ecl.getOutDim();
        } else {
            inputCount = dil.getOutDim();
        }
        double beta = 0.7 * Math.pow(hiddenCount, 1.0 / inputCount);

        for(Layer layer: this.hiddenLayers) {
            if(!(layer instanceof DenseLayer)) {
                continue;
            }
            initDenserLayerWeights((DenseLayer) layer, beta);
        }

        initDenserLayerWeights(finalLayer, beta);

        if(this.wideEnable && this.deepEnable) {
            initDenserLayerWeights(wdLayer, beta);
        }

        // TODO init embed layers, does beta value need to be changed?
        LOG.info("Init weight be called with mode:{}", defaultMode.name());
    }

    private void initDenserLayerWeights(DenseLayer layer, double beta) {
        double n = 0d;
        double[][] weights = layer.getWeights();
        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                n += (weights[i][j] * weights[i][j]);
            }
        }
        double[] bias = layer.getBias();
        for(int i = 0; i < bias.length; i++) {
            n += (bias[i] * bias[i]);
        }
        n = Math.sqrt(n);

        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                weights[i][j] = beta * weights[i][j] / n;
            }
        }
        for(int i = 0; i < bias.length; i++) {
            bias[i] = beta * bias[i] / n;
        }
    }

    @SuppressWarnings("rawtypes")
    @Override
    public void initWeight(InitMethod method) {
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer.initWeight(method);
        }

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

    @Override
    public void initWeight(WideAndDeep updateModel) {
        AssertUtils.assertListNotNullAndSizeEqual(this.hiddenLayers, updateModel.getHiddenLayers());
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(this.hiddenLayers.get(i) instanceof DenseLayer) {
                ((DenseLayer) this.hiddenLayers.get(i)).initWeight((DenseLayer) updateModel.getHiddenLayers().get(i));
            }
        }
        this.finalLayer.initWeight(updateModel.getFinalLayer());
        this.ecl.initWeight(updateModel.getEcl());
        this.wl.initWeight(updateModel.getWl());
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer.initWeight(updateModel.wdLayer);
        }
    }

    public void write(DataOutput out, SerializationType serializationType) throws IOException {
        this.serializationType = serializationType;
        write(out);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.serializationType.getValue());
        out.writeBoolean(this.isWideEnable());
        out.writeBoolean(this.isDeepEnable());
        out.writeBoolean(this.isEmbedEnable());
        out.writeBoolean(this.isWideDenseEnable());

        writeLayerWithNuLLCheck(out, this.dil);

        if(this.hiddenLayers == null) {
            out.writeInt(NULL);
        } else {
            List<DenseLayer> denseLayers = this.hiddenLayers.stream().filter(layer -> layer instanceof DenseLayer)
                    .map(layer -> (DenseLayer) layer).collect(Collectors.toList());
            out.writeInt(denseLayers.size());

            denseLayers.forEach(denseLayer -> {
                try {
                    denseLayer.write(out, this.serializationType);
                } catch (IOException e) {
                    LOG.error("IOException when write hidden nodes dense part", e);
                }
            });
        }

        writeLayerWithNuLLCheck(out, finalLayer);
        writeLayerWithNuLLCheck(out, ecl);
        writeLayerWithNuLLCheck(out, wl);
        if(this.wideEnable && this.deepEnable) {
            writeLayerWithNuLLCheck(out, this.wdLayer);
        }

        if(this.actiFuncs == null) {
            out.writeInt(NULL);
        } else {
            out.writeInt(this.actiFuncs.size());
            this.actiFuncs.forEach(act -> {
                try {
                    out.writeUTF(act);
                } catch (IOException e) {
                    LOG.error("Write active function " + act, e);
                }
            });
        }

        if(this.serializationType == SerializationType.MODEL_SPEC) {
            if(idBinCateSizeMap == null) {
                out.writeInt(NULL);
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
            out.writeDouble(l2reg);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        this.serializationType = SerializationType.getSerializationType(in.readInt());
        this.wideEnable = in.readBoolean();
        this.deepEnable = in.readBoolean();
        this.embedEnable = in.readBoolean();
        this.wideDenseEnable = in.readBoolean();

        this.dil = (DenseInputLayer) readLayerWithNullCheck(in, new DenseInputLayer());

        List<DenseLayer> hiddenDenseLayer = new ArrayList<>();
        int size = in.readInt();
        for(int i = 0; i < size; i++) {
            DenseLayer denseLayer = new DenseLayer();
            denseLayer.readFields(in, this.serializationType);
            hiddenDenseLayer.add(denseLayer);
        }

        this.finalLayer = (DenseLayer) readLayerWithNullCheck(in, new DenseLayer());
        this.ecl = (EmbedLayer) readLayerWithNullCheck(in, new EmbedLayer());
        this.wl = (WideLayer) readLayerWithNullCheck(in, new WideLayer());
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer = (DenseLayer) readLayerWithNullCheck(in, new DenseLayer());
        }

        this.actiFuncs = new ArrayList<>();
        size = in.readInt();
        for(int i = 0; i < size; i++) {
            this.actiFuncs.add(in.readUTF());
        }

        AssertUtils.assertListNotNullAndSizeEqual(this.actiFuncs, hiddenDenseLayer);
        // hiddenDenseLayer.forEach(denseLayer -> LOG.info(String.valueOf(denseLayer)));
        this.hiddenLayers = new ArrayList<>(this.actiFuncs.size() * 2);
        for(int i = 0; i < hiddenDenseLayer.size(); i++) {
            this.hiddenLayers.add(hiddenDenseLayer.get(i));
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(this.actiFuncs.get(i)));
        }

        if(serializationType == SerializationType.MODEL_SPEC) {
            size = in.readInt();
            this.idBinCateSizeMap = new HashMap<>(size);
            for(int i = 0; i < size; i++) {
                idBinCateSizeMap.put(in.readInt(), in.readInt());
            }
            numericalSize = in.readInt();
            denseColumnIds = SerializationUtil.readIntList(in, denseColumnIds);
            embedColumnIds = SerializationUtil.readIntList(in, embedColumnIds);
            embedOutputs = SerializationUtil.readIntList(in, embedOutputs);
            wideColumnIds = SerializationUtil.readIntList(in, wideColumnIds);
            hiddenNodes = SerializationUtil.readIntList(in, hiddenNodes);
            l2reg = in.readDouble();
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
            layer.readFields(in, this.serializationType);
        }
        return layer;
    }

    @SuppressWarnings("rawtypes")
    @Override
    public WideAndDeep combine(WideAndDeep from) {
        this.dil = this.dil.combine(from.getDil());

        List<Layer> fhl = from.getHiddenLayers();
        int hlSize = hiddenLayers.size();
        List<Layer> combinedLayers = new ArrayList<Layer>(hlSize);
        for(int i = 0; i < hlSize; i++) {
            if(hiddenLayers.get(i) instanceof DenseLayer) {
                Layer nLayer = ((DenseLayer) hiddenLayers.get(i)).combine((DenseLayer) fhl.get(i));
                combinedLayers.add(nLayer);
            } else {
                combinedLayers.add(hiddenLayers.get(i));
            }
        }
        this.hiddenLayers = combinedLayers;

        this.finalLayer = this.finalLayer.combine(from.getFinalLayer());
        this.ecl = this.ecl.combine(from.getEcl());
        this.wl = this.wl.combine(from.getWl());
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer = this.wdLayer.combine(from.wdLayer);
        }
        return this;
    }

    @SuppressWarnings("rawtypes")
    public void update(WideAndDeep gradWnd, Optimizer optimizer, double trainCount) {
        this.dil.update(gradWnd.getDil(), optimizer, StringUtils.EMPTY, trainCount);

        List<Layer> gradHLs = gradWnd.getHiddenLayers();
        int hlSize = hiddenLayers.size();
        for(int i = 0; i < hlSize; i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if(tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).update((DenseLayer) gradHLs.get(i), optimizer, "h" + i, trainCount);
            }
        }

        this.finalLayer.update(gradWnd.getFinalLayer(), optimizer, "f", trainCount);
        this.ecl.update(gradWnd.getEcl(), optimizer, "e", trainCount);
        this.wl.update(gradWnd.getWl(), optimizer, "w", trainCount);
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer.update(gradWnd.wdLayer, optimizer, "wd", trainCount);
        }
    }

    /**
     * @return the index
     */
    public int getIndex() {
        return index;
    }

    /**
     * @param index
     *            the index to set
     */
    public void setIndex(int index) {
        this.index = index;
    }

    @SuppressWarnings("rawtypes")
    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        for(Layer layer: this.hiddenLayers) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initOptimizer(learningRate, algorithm, reg, rl);
            }
        }
        this.finalLayer.initOptimizer(learningRate, algorithm, reg, rl);
        this.ecl.initOptimizer(learningRate, algorithm, reg, rl);
        this.wl.initOptimizer(learningRate, algorithm, reg, rl);
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer.initOptimizer(learningRate, algorithm, reg, rl);
        }
    }

    @SuppressWarnings("rawtypes")
    @Override
    public void optimizeWeight(double numTrainSize, int iteration, WideAndDeep gradWnd) {
        List<Layer> gradHLs = gradWnd.getHiddenLayers();
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if(tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).optimizeWeight(numTrainSize, iteration, (DenseLayer) gradHLs.get(i));
            }
        }

        this.finalLayer.optimizeWeight(numTrainSize, iteration, gradWnd.getFinalLayer());
        this.ecl.optimizeWeight(numTrainSize, iteration, gradWnd.getEcl());
        this.wl.optimizeWeight(numTrainSize, iteration, gradWnd.getWl());
        if(this.wideEnable && this.deepEnable) {
            this.wdLayer.optimizeWeight(numTrainSize, iteration, gradWnd.wdLayer);

        }
    }

    @Override
    public WideAndDeep clone() {
        // Set the initial buffer size to 1M
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(1024 * 1024);
        DataOutputStream dos = new DataOutputStream(byteArrayOutputStream);
        DataInputStream dis = null;
        try {
            write(dos, SerializationType.MODEL_SPEC);
            dos.flush();
            ByteArrayInputStream dataInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
            WideAndDeep wideAndDeep = new WideAndDeep();
            dis = new DataInputStream(dataInputStream);
            wideAndDeep.readFields(dis);
            wideAndDeep.initGrads();
            return wideAndDeep;
        } catch (IOException e) {
            LOG.error("IOException happen when clone wideAndDeep model", e);
        } finally {
            IOUtils.closeStream(dos);
            if(dis != null) {
                IOUtils.closeStream(dis);
            }
        }
        return null;
    }

    /**
     * @return the wideEnable
     */
    public boolean isWideEnable() {
        return wideEnable;
    }

    /**
     * @param wideEnable
     *            the wideEnable to set
     */
    public void setWideEnable(boolean wideEnable) {
        this.wideEnable = wideEnable;
    }

    /**
     * @return the deepEnable
     */
    public boolean isDeepEnable() {
        return deepEnable;
    }

    /**
     * @param deepEnable
     *            the deepEnable to set
     */
    public void setDeepEnable(boolean deepEnable) {
        this.deepEnable = deepEnable;
    }

    /**
     * @return the embedEnable
     */
    public boolean isEmbedEnable() {
        return embedEnable;
    }

    /**
     * @param embedEnable
     *            the embedEnable to set
     */
    public void setEmbedEnable(boolean embedEnable) {
        this.embedEnable = embedEnable;
    }

    /**
     * @return the wideDenseEnable
     */
    public boolean isWideDenseEnable() {
        return wideDenseEnable;
    }

    /**
     * @param wideDenseEnable
     *            the wideDenseEnable to set
     */
    public void setWideDenseEnable(boolean wideDenseEnable) {
        this.wideDenseEnable = wideDenseEnable;
    }

}
