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
package ml.shifu.shifu.core.dtrain.mtl;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.core.dtrain.wdl.*;

import ml.shifu.shifu.core.dtrain.wdl.activation.Activation;
import ml.shifu.shifu.core.dtrain.wdl.activation.ActivationFactory;
import ml.shifu.shifu.core.dtrain.wdl.activation.Sigmoid;
import ml.shifu.shifu.core.dtrain.wdl.optimization.PropOptimizer;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static ml.shifu.shifu.core.dtrain.wdl.SerializationUtil.NULL;

/**
 * @author haillu
 */
public class MultiTaskLearning implements WeightInitializer<MultiTaskLearning>, Bytable, Combinable<MultiTaskLearning>,
        PropOptimizer<MultiTaskLearning> {

    private static final Logger LOG = LoggerFactory.getLogger(MultiTaskLearning.class);

    private DenseInputLayer dil;

    private int inputSize;

    private List<Layer> hiddenLayers;

    private List<Integer> hiddenNodes;

    private List<String> hiddenActiFuncs;

    private DenseLayer finalLayer;

    private int taskNumber;

    // todo: different tasks should have the activation func respectively.
    // private List<String> finalActiFuncs;
    private Activation finalActiFunc;

    private double l2reg;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    public MultiTaskLearning() {
    }

    public MultiTaskLearning(int inputSize, List<Integer> hiddenNodes, List<String> HiddenActiFuncs, int taskNumber,
            double l2reg) {
        this.inputSize = inputSize;
        this.hiddenNodes = hiddenNodes;
        this.hiddenActiFuncs = HiddenActiFuncs;
        this.taskNumber = taskNumber;
        // this.finalActiFuncs = finalActiFuncs;
        this.l2reg = l2reg;

        // build the structure of NN graph.
        this.dil = new DenseInputLayer(this.inputSize);
        int preHiddenInputs = dil.getOutDim();

        AssertUtils.assertListNotNullAndSizeEqual(hiddenNodes, HiddenActiFuncs);
        this.hiddenLayers = new ArrayList<>(hiddenNodes.size() * 2);
        for(int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs, l2reg);
            this.hiddenLayers.add(denseLayer);
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(HiddenActiFuncs.get(i)));
            preHiddenInputs = hiddenOutputs;
        }

        // AssertUtils.assertEquals(taskNumber, finalActiFuncs.size());
        this.finalLayer = new DenseLayer(taskNumber, preHiddenInputs, l2reg);
        // Use Sigmoid as final actiFunc of all tasks.
        this.finalActiFunc = new Sigmoid();
    }

    public double[] forward(double[] denseInputs) {
        // input layer forward
        double[] dilOuts = this.dil.forward(denseInputs);
        double[] inputs = dilOuts;
        // hidden layers forward
        for(Layer layer: this.hiddenLayers) {
            if(layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                inputs = denseLayer.forward(inputs);
            } else if(layer instanceof Activation) {
                Activation acti = (Activation) layer;
                inputs = acti.forward(inputs);
            }
        }
        // final layer forward
        inputs = this.finalLayer.forward(inputs);
        // TODO:final Activation list.
        double[] resluts = this.finalActiFunc.forward(inputs);
        return resluts;
    }

    public float[] backward(double[] predicts, double[] actuals, double sig) {
        // replace the finalActiFunc layer.
        double[] grad2Logits = new double[predicts.length];

        for(int i = 0; i < grad2Logits.length; i++) {
            grad2Logits[i] = (predicts[i] - actuals[i]) * (predicts[i] * (1 - predicts[i])) * sig;
            // error * sigmoid derivertive * weight
        }

        double[] backInputs = this.finalLayer.backward(grad2Logits);
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1 - i);
            if(layer instanceof DenseLayer) {
                backInputs = ((DenseLayer) layer).backward(backInputs);
            } else if(layer instanceof Activation) {
                backInputs = ((Activation) layer).backward(backInputs);
            }
        }

        // no need return final backward outputs as gradients are computed well
        return null;
    }

    /**
     * Initialize gradients for training of each epoch
     */
    public void initGrads() {
        for(Layer layer: hiddenLayers) {
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initGrads();
            }
        }
        this.finalLayer.initGrads();
    }

    public void updateWeights(MultiTaskLearning multiTask) {
        this.initWeight(multiTask);
    }

    public void updateWeights(MTLParams params) {
        updateWeights(params.getMtl());
        // after update weights, gradients should be re newed.
        this.initGrads();
    }

    /**
     * Init the weights in MultiTaskLearning Model and it's sub module
     */
    public void initWeights() {
        InitMethod defaultMode = InitMethod.ZERO_ONE_RANGE_RANDOM;
        initWeight(defaultMode);
        LOG.error("Init weight be called with mode:" + defaultMode.name());
    }

    @Override
    public void initWeight(InitMethod method) {
        for(Layer layer: this.hiddenLayers) {
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight(method);
            }
        }
        this.finalLayer.initWeight(method);
    }

    @Override
    public void initWeight(MultiTaskLearning updateModel) {
        AssertUtils.assertListNotNullAndSizeEqual(this.hiddenLayers, updateModel.getHiddenLayers());
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(i);
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight((DenseLayer) updateModel.getHiddenLayers().get(i));
            }
            this.finalLayer.initWeight(updateModel.getFinalLayer());
        }
    }

    public void write(DataOutput out, SerializationType serializationType) throws IOException {
        this.serializationType = serializationType;
        write(out);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.serializationType.getValue());
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

        if(this.hiddenActiFuncs == null) {
            out.writeInt(NULL);
        } else {
            out.writeInt(this.hiddenActiFuncs.size());
            this.hiddenActiFuncs.forEach(act -> {
                try {
                    out.writeUTF(act);
                } catch (IOException e) {
                    LOG.error("Write active function " + act, e);
                }
            });
        }

        writeLayerWithNuLLCheck(out, finalLayer);
        out.writeUTF(this.finalActiFunc.getClass().getSimpleName());
        if(this.serializationType == SerializationType.MODEL_SPEC) {
            out.writeInt(inputSize);
            out.writeDouble(l2reg);
        }
    }

    // used in method 'readFields' for local tests .
    public int findPos(Object dis) throws NoSuchFieldException, IllegalAccessException {
        Class clazz = dis.getClass().getSuperclass();
        Field field = clazz.getDeclaredField("in");
        field.setAccessible(true);
        ByteArrayInputStream byteIS = (ByteArrayInputStream) field.get(dis);

        clazz = byteIS.getClass();
        field = clazz.getDeclaredField("pos");
        field.setAccessible(true);
        int pos = (int) field.get(byteIS);
        System.out.println("pos:" + pos);
        return pos;
    }

    @Override
    public void readFields(DataInput in) throws IOException {

        // test the pos of inputStream
        // try {
        // findPos(in);
        // } catch (NoSuchFieldException e) {
        // e.printStackTrace();
        // } catch (IllegalAccessException e) {
        // e.printStackTrace();
        // }

        this.serializationType = SerializationType.getSerializationType(in.readInt());
        this.dil = (DenseInputLayer) readLayerWithNullCheck(in, new DenseInputLayer());
        List<DenseLayer> hiddenDenseLayer = new ArrayList<>();
        int size = in.readInt();
        for(int i = 0; i < size; i++) {
            DenseLayer denseLayer = new DenseLayer();
            denseLayer.readFields(in, this.serializationType);
            hiddenDenseLayer.add(denseLayer);
        }

        this.hiddenActiFuncs = new ArrayList<>();
        size = in.readInt();
        for(int i = 0; i < size; i++) {
            this.hiddenActiFuncs.add(in.readUTF());
        }
        this.finalLayer = (DenseLayer) readLayerWithNullCheck(in, new DenseLayer());
        this.finalActiFunc = ActivationFactory.getInstance().getActivation(in.readUTF());

        // build hiddenLayers including activations:
        AssertUtils.assertListNotNullAndSizeEqual(this.hiddenActiFuncs, hiddenDenseLayer);
        this.hiddenLayers = new ArrayList<>(this.hiddenActiFuncs.size() * 2);
        for(int i = 0; i < hiddenDenseLayer.size(); i++) {
            this.hiddenLayers.add(hiddenDenseLayer.get(i));
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(this.hiddenActiFuncs.get(i)));
        }

        if(serializationType == SerializationType.MODEL_SPEC) {
            this.inputSize = in.readInt();
            this.l2reg = in.readDouble();
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

    @Override
    public MultiTaskLearning combine(MultiTaskLearning from) {
        // combine input layers.
        this.dil = this.dil.combine(from.getDil());

        // combine hidden layers
        List<Layer> fhl = from.hiddenLayers;
        int hlSize = hiddenLayers.size();
        List<Layer> combinedLayers = new ArrayList<Layer>(hlSize);
        for(int i = 0; i < hlSize; i++) {
            if(hiddenLayers.get(i) instanceof DenseLayer) {
                Layer nLayer = ((DenseLayer) hiddenLayers.get(i)).combine((DenseLayer) fhl.get(i));
                combinedLayers.add(nLayer);
            }
            // just copy activation layers from mtl before.
            else {
                combinedLayers.add(hiddenLayers.get(i));
            }
        }
        this.hiddenLayers = combinedLayers;

        this.finalLayer = this.finalLayer.combine(from.getFinalLayer());

        this.finalActiFunc = from.finalActiFunc;
        return this;
    }

    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        for(Layer layer: this.hiddenLayers) {
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initOptimizer(learningRate, algorithm, reg, rl);
            }
        }
        this.finalLayer.initOptimizer(learningRate, algorithm, reg, rl);
    }

    @Override
    public void optimizeWeight(double numTrainSize, int iteration, MultiTaskLearning model) {
        List<Layer> gradHLs = model.getHiddenLayers();
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if(tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).optimizeWeight(numTrainSize, iteration, (DenseLayer) gradHLs.get(i));
            }
        }
        this.finalLayer.optimizeWeight(numTrainSize, iteration, model.getFinalLayer());
    }

    // clone by serialization
    @Override
    public MultiTaskLearning clone() {
        // Set the initial buffer size to 1M
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(1024 * 1024);
        DataOutputStream dos = new DataOutputStream(byteArrayOutputStream);
        DataInputStream dis = null;
        try {
            write(dos, SerializationType.MODEL_SPEC);
            dos.flush();
            ByteArrayInputStream dataInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
            MultiTaskLearning mtl = new MultiTaskLearning();
            dis = new DataInputStream(dataInputStream);
            mtl.readFields(dis);
            mtl.initGrads();
            return mtl;
        } catch (IOException e) {
            LOG.error("IOException happen when clone mtl model", e);
        } finally {
            IOUtils.closeStream(dos);
            if(dis != null) {
                IOUtils.closeStream(dis);
            }
        }
        return null;
    }

    public DenseInputLayer getDil() {
        return dil;
    }

    public void setDil(DenseInputLayer dil) {
        this.dil = dil;
    }

    public List<Layer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(List<Layer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public DenseLayer getFinalLayer() {
        return finalLayer;
    }

    public void setFinalLayer(DenseLayer finalLayer) {
        this.finalLayer = finalLayer;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public List<Integer> getHiddenNodes() {
        return hiddenNodes;
    }

    public void setHiddenNodes(List<Integer> hiddenNodes) {
        this.hiddenNodes = hiddenNodes;
    }

    public List<String> getHiddenActiFuncs() {
        return hiddenActiFuncs;
    }

    public void setHiddenActiFuncs(List<String> hiddenActiFuncs) {
        this.hiddenActiFuncs = hiddenActiFuncs;
    }

    public double getL2reg() {
        return l2reg;
    }

    public void setL2reg(double l2reg) {
        this.l2reg = l2reg;
    }

    public SerializationType getSerializationType() {
        return serializationType;
    }

    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    public int getTaskNumber() {
        return taskNumber;
    }

    public void setTaskNumber(int taskNumber) {
        this.taskNumber = taskNumber;
    }
}
