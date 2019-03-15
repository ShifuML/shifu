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
import static ml.shifu.shifu.core.dtrain.wdl.SerializationUtil.NULL;
import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;
import ml.shifu.shifu.util.Tuple;

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
        extends AbstractLayer<Tuple<List<SparseInput>, float[]>, float[], float[], List<float[]>, WideLayer>
        implements WeightInitializer {

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

    @Override
    public float[] forward(Tuple<List<SparseInput>, float[]> input) {
        AssertUtils.assertListNotNullAndSizeEqual(this.getLayers(), input.getFirst());
        float[] results = new float[layers.get(0).getOutDim()];
        for(int i = 0; i < getLayers().size(); i++) {
            float[] fOuts = this.getLayers().get(i).forward(input.getFirst().get(i));
            for(int j = 0; j < results.length; j++) {
                results[j] += fOuts[j];
            }
        }

        float[] denseForwards = this.denseLayer.forward(input.getSecond());
        assert denseForwards.length == results.length;
        for(int j = 0; j < results.length; j++) {
            results[j] += denseForwards[j];
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

        list.add(this.denseLayer.backward(backInputs, sig));
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

    @Override
    public void initWeight(InitMethod method) {
        for(WideFieldLayer layer: this.layers) {
            layer.initWeight(method);
        }
        this.denseLayer.initWeight(method);
        this.bias.initWeight(method);
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
    public void update(WideLayer gradLayer, Optimizer optimizer) {
        List<WideFieldLayer> gradWFLs = gradLayer.getLayers();
        int wflSize = this.layers.size();
        for(int i = 0; i < wflSize; i++) {
            this.layers.get(i).update(gradWFLs.get(i), optimizer);
        }
        this.denseLayer.update(gradLayer.getDenseLayer(), optimizer);
        this.bias.update(gradLayer.getBias(), optimizer);
    }

}
