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

import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import static ml.shifu.shifu.core.dtrain.wdl.SerializationUtil.NULL;
import ml.shifu.shifu.core.dtrain.wdl.optimization.PropOptimizer;
import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;
import ml.shifu.shifu.util.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        extends AbstractLayer<Tuple<List<SparseInput>, double[]>, double[], double[], List<double[]>, WideLayer>
        implements WeightInitializer<WideLayer>, PropOptimizer<WideLayer> {
    @SuppressWarnings("unused")
    private static final Logger LOG = LoggerFactory.getLogger(WideLayer.class);
    /**
     * Layers for all wide columns.
     */
    private List<WideFieldLayer> layers;

    /**
     * Layers for all wide columns.
     */
    private WideDenseLayer denseLayer;

    /**
     * Bias layer
     */
    private BiasLayer bias;

    public WideLayer() {
    }

    public WideLayer(List<WideFieldLayer> layers, BiasLayer bias) {
        this.layers = layers;
        this.bias = bias;
    }

    public WideLayer(List<WideFieldLayer> layers, WideDenseLayer denseLayer, BiasLayer bias) {
        this.layers = layers;
        this.bias = bias;
        this.denseLayer = denseLayer;
    }

    @Override
    public int getOutDim() {
        int len = 0;
        for(WideFieldLayer layer: getLayers()) {
            len += layer.getOutDim();
        }
        len += 1; // bias
        len += 1; // WideDenseLayer
        return len;
    }

    private boolean isDebug = false;

    @Override
    public double[] forward(Tuple<List<SparseInput>, double[]> input) {
//        LOG.debug("Debug in Wide Layer: with input first " + input.getFirst().size() + " second "
//                + input.getSecond().length);
        AssertUtils.assertListNotNullAndSizeEqual(this.getLayers(), input.getFirst());
        double[] results = new double[layers.get(0).getOutDim()];
        for(int i = 0; i < getLayers().size(); i++) {
            double[] fOuts = this.getLayers().get(i).forward(input.getFirst().get(i));
            for(int j = 0; j < results.length; j++) {
//                if(this.isDebug) {
//                    LOG.debug("outputs " + j + " value is " + fOuts[j]);
//                }
                results[j] += fOuts[j];
            }
        }

//        this.denseLayer.setDebug(isDebug);
        double[] denseForwards = this.denseLayer.forward(input.getSecond());
//        LOG.debug("Densor forward:");
        assert denseForwards.length == results.length;
        for(int j = 0; j < results.length; j++) {
//            if(this.isDebug) {
//                LOG.info("Densor forward " + j + " value is " + denseForwards[j]);
//            }
            results[j] += denseForwards[j];
        }

        for(int j = 0; j < results.length; j++) {
//            if(this.isDebug) {
//                LOG.debug("before add bias result " + j + " is " + results[j]);
//            }
            results[j] += bias.forward(1d);
            // if(this.isDebug) {
            // LOG.debug("after add bias result " + j + " is " + results[j]);
            // }
        }
        return results;
    }

    @Override
    public List<double[]> backward(double[] backInputs) {
        // below backward call is for gradients computation in WideFieldLayer and BiasLayer
        List<double[]> list = new ArrayList<>();
        for(int i = 0; i < getLayers().size(); i++) {
            list.add(this.getLayers().get(i).backward(backInputs));
        }

        list.add(this.denseLayer.backward(backInputs));
        list.add(new double[] { bias.backward(backInputs[0]) });
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

    @Override
    public void initWeight(InitMethod method) {
        for(WideFieldLayer layer: this.layers) {
            layer.initWeight(method);
        }
        this.denseLayer.initWeight(method);
        this.bias.initWeight(method);
    }

    @Override
    public void initWeight(WideLayer updateModel) {
        AssertUtils.assertListNotNullAndSizeEqual(this.layers, updateModel.getLayers());
        for(int i = 0; i < this.layers.size(); i++) {
            this.layers.get(i).initWeight(updateModel.getLayers().get(i));
        }
        this.denseLayer.initWeight(updateModel.getDenseLayer());
        this.bias.initWeight(updateModel.getBias());
    }

    public void initGrads() {
        for(WideFieldLayer layer: this.layers) {
            layer.initGrads();
        }
        this.denseLayer.initGrads();
        this.bias.initGrads();
    }

    /**
     * @return the denseLayer
     */
    public WideDenseLayer getDenseLayer() {
        return denseLayer;
    }

    /**
     * @param denseLayer
     *            the denseLayer to set
     */
    public void setDenseLayer(WideDenseLayer denseLayer) {
        this.denseLayer = denseLayer;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        if(this.layers == null) {
            out.writeInt(NULL);
        } else {
            out.writeInt(this.layers.size());
            for(WideFieldLayer wideFieldLayer: this.layers) {
                wideFieldLayer.write(out, this.serializationType);
            }
        }

        if(this.denseLayer == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            this.denseLayer.write(out, this.serializationType);
        }

        if(this.bias == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            this.bias.write(out, this.serializationType);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        int layerSize = in.readInt();
        this.layers = new ArrayList<>(layerSize);
        for(int i = 0; i < layerSize; i++) {
            WideFieldLayer wideFieldLayer = new WideFieldLayer();
            wideFieldLayer.readFields(in, this.serializationType);
            this.layers.add(wideFieldLayer);
        }

        if(in.readBoolean()) {
            this.denseLayer = new WideDenseLayer();
            this.denseLayer.readFields(in, this.serializationType);
        }

        if(in.readBoolean()) {
            this.bias = new BiasLayer();
            this.bias.readFields(in, this.serializationType);
        }
    }

    @Override
    public WideLayer combine(WideLayer from) {
        List<WideFieldLayer> fLayers = from.getLayers();
        int wflSize = this.layers.size();
        List<WideFieldLayer> combinedLayers = new ArrayList<WideFieldLayer>(wflSize);
        for(int i = 0; i < wflSize; i++) {
            WideFieldLayer nLayer = layers.get(i).combine(fLayers.get(i));
            combinedLayers.add(nLayer);
        }
        this.layers = combinedLayers;

        denseLayer = denseLayer.combine(from.getDenseLayer());
        bias = bias.combine(from.getBias());
        return this;
    }

    @Override
    public void update(WideLayer gradLayer, Optimizer optimizer, String uniqueKey, double trainCount) {
        List<WideFieldLayer> gradWFLs = gradLayer.getLayers();
        int wflSize = this.layers.size();
        for(int i = 0; i < wflSize; i++) {
            this.layers.get(i).update(gradWFLs.get(i), optimizer, uniqueKey + "w" + i, trainCount);
        }
        this.denseLayer.update(gradLayer.getDenseLayer(), optimizer, "d", trainCount);
        this.bias.update(gradLayer.getBias(), optimizer, "b", trainCount);
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

    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        for(WideFieldLayer wideFieldLayer: this.layers) {
            wideFieldLayer.initOptimizer(learningRate, algorithm, reg, rl);
        }
        this.denseLayer.initOptimizer(learningRate, algorithm, reg, rl);
    }

    @Override
    public void optimizeWeight(double numTrainSize, int iteration, WideLayer model) {
        for(int i = 0; i < this.layers.size(); i++) {
            this.layers.get(i).optimizeWeight(numTrainSize, iteration, model.getLayers().get(i));
        }
        this.denseLayer.optimizeWeight(numTrainSize, iteration, model.getDenseLayer());
    }
}
