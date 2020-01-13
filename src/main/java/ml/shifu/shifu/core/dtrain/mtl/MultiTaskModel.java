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

import static ml.shifu.shifu.core.dtrain.layer.SerializationUtil.NULL;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.layer.AbstractLayer;
import ml.shifu.shifu.core.dtrain.layer.DenseInputLayer;
import ml.shifu.shifu.core.dtrain.layer.DenseLayer;
import ml.shifu.shifu.core.dtrain.layer.Layer;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;
import ml.shifu.shifu.core.dtrain.layer.SerializationUtil;
import ml.shifu.shifu.core.dtrain.layer.WeightInitializer;
import ml.shifu.shifu.core.dtrain.layer.WeightInitializer.InitMethod;
import ml.shifu.shifu.core.dtrain.layer.activation.Activation;
import ml.shifu.shifu.core.dtrain.layer.activation.ActivationFactory;
import ml.shifu.shifu.core.dtrain.layer.optimization.PropOptimizer;
import ml.shifu.shifu.util.CommonUtils;

/**
 * {@link MultiTaskModel} is a memory model defined for multiple task model structure. And can also be used in gradients
 * aggregation. Message body in master-workers distributed epoch iteration.
 * 
 * <p>
 * The model architecture is a typical multiple task model includes input layer, fully connected hidden layers and
 * multiple final output layers. In this architecture, only last final output layers are independent layers for
 * different tasks. Hidden layers are all shared and would be shared trained.
 * 
 * <p>
 * {@link #serializationType} is used to denote if such model is for model spec loading from model file, if it is master
 * models which updated in each epoch or worker gradients accumulated to master for final model update.
 * 
 * <P>
 * {@link Combinable} is implemented for better gradients aggregation in master which can improve memory consumption and
 * aggregation efficiency. {@link PropOptimizer} is to define gradient descent or adam like optimizer for final model
 * updates in each epoch.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MultiTaskModel implements WeightInitializer<MultiTaskModel>, Bytable, Combinable<MultiTaskModel>,
        PropOptimizer<MultiTaskModel> {

    private static final Logger LOG = LoggerFactory.getLogger(MultiTaskModel.class);

    /**
     * Dense input layer, multi inputs are accumulated into one double array and set to such layer.
     */
    private DenseInputLayer dil;

    /**
     * Hidden layers which are shared by all multiple tasks.
     */
    @SuppressWarnings("rawtypes")
    private List<Layer> hiddenLayers;

    /**
     * Final dense layers, each one is for each task.
     */
    private List<DenseLayer> finalLayers;

    /**
     * L2 regularization used for gradient descent training.
     */
    private double l2reg;

    /**
     * Serialization type for gradients, weights or model spec.
     */
    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    /**
     * Hidden nodes defined in hidden layer.
     */
    private List<Integer> hiddenNodes;

    /**
     * Activiation functions defined in hidden layers.
     */
    private List<String> actiFuncs;

    /**
     * Final output dimensions (each one for one task).
     */
    private List<Integer> finalOutputs;

    /**
     * Flat spot value to smooth lr derived function: result * (1 - result): This value sometimes may be close to zero.
     * Add flat sport to improve it: result * (1 - result) + 0.1d
     */
    private static final double FLAT_SPOT_VALUE = 0.1d;

    /**
     * Default constructor which is for de-serialization.
     */
    public MultiTaskModel() {
    }

    /**
     * Constructor from user defined input, outputs and hidden layer settings.
     * 
     * @param input
     *            number of input array
     * @param hiddenNodes
     *            number of hidden nodes in hidden layers
     * @param actiFuncs
     *            activiation function in each hidden layer
     * @param finalOutputs
     *            number of outputs in all tasks
     * @param l2reg
     *            the L2 regularization used for gradient descent training
     */
    public MultiTaskModel(int input, List<Integer> hiddenNodes, List<String> actiFuncs, List<Integer> finalOutputs,
            double l2reg) {
        this.dil = new DenseInputLayer(input);
        this.hiddenNodes = hiddenNodes;
        this.actiFuncs = actiFuncs;
        this.finalOutputs = finalOutputs;
        this.l2reg = l2reg;

        int preHiddenInputs = input;
        AssertUtils.assertListNotNullAndSizeEqual(hiddenNodes, actiFuncs);
        this.setHiddenLayers(new ArrayList<>(hiddenNodes.size() * 2));
        for(int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs, l2reg);
            this.hiddenLayers.add(denseLayer);
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(actiFuncs.get(i)));
            preHiddenInputs = hiddenOutputs;
        }

        AssertUtils.assertNotNull(finalOutputs);
        this.finalLayers = new ArrayList<>(finalOutputs.size());
        for(int i = 0; i < finalOutputs.size(); i++) {
            // 0-1 classification per each task, output = 1
            this.finalLayers.add(new DenseLayer(1, preHiddenInputs, l2reg));
        }
    }

    /**
     * Update model weights from extra {@link MultiTaskModel} instance.
     * 
     * @param mtm
     *            the model to update local weights
     */
    public void updateWeights(MultiTaskModel mtm) {
        this.initWeight(mtm);
    }

    /**
     * Update model weights from extra {@link MTLParams} instance, as weights are updated, gradients has been reset for
     * next epoch.
     * 
     * @param params
     *            the params with weights to update local model
     */
    public void updateWeights(MTLParams params) {
        updateWeights(params.getMtm());
        // after update weights, gradients should be re newed.
        this.initGrads();
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
        for(int j = 0; j < finalLayers.size(); j++) {
            this.finalLayers.get(j).initGrads();
        }
    }

    /**
     * Logits forward computation from input layer -&gt; hidden layers -&gt; output layers. Output layers are not shared
     * but
     * others are shared in multiple tasks.
     * 
     * @param denseInputs
     *            the dense inputs accumulated from different input sources.
     * @return logits of final outputs
     */
    @SuppressWarnings("rawtypes")
    public double[] forward(double[] denseInputs) {
        double[] dilOuts = this.dil.forward(denseInputs);
        double[] forwards = dilOuts;
        for(Layer layer: this.hiddenLayers) {
            if(layer instanceof DenseLayer) {
                DenseLayer dl = (DenseLayer) layer;
                forwards = dl.forward(forwards);
            } else if(layer instanceof Activation) {
                Activation acti = (Activation) layer;
                forwards = acti.forward(forwards);
            }
        }

        double[] results = new double[this.finalLayers.size()];
        for(int j = 0; j < finalLayers.size(); j++) {
            double[] outputs = finalLayers.get(j).forward(forwards);
            results[j] = outputs[0];
        }
        return results;
    }

    /**
     * Backward computation to get gradients. Gradients are computed and saved in each layer based on backward errors.
     * 
     * @param predicts
     *            predicted values
     * @param actuals
     *            the actual target values
     * @param sig
     *            weighted parameter like dollar
     * @return null as gradients are accumulated in backward computation of each layer.
     */
    @SuppressWarnings("rawtypes")
    public double[] backward(double[] predicts, double[] actuals, double sig) {
        // TODO add binary cross entropy here
        // 1. Error computation based on outputs of different target.
        double[] grad2Logits = new double[predicts.length];
        for(int i = 0; i < grad2Logits.length; i++) {
            double error = (predicts[i] - actuals[i]);
            grad2Logits[i] = error * (CommonUtils.sigmoidDerivedFunction(predicts[i]) + FLAT_SPOT_VALUE) * sig * -1d;
        }

        // 2. Final layers backward accumulation
        double[] backInputs = new double[finalLayers.get(0).getIn()];
        for(int i = 0; i < this.finalLayers.size(); i++) {
            backInputs = CommonUtils.plus(backInputs,
                    this.finalLayers.get(i).backward(new double[] { grad2Logits[i] }));
        }

        // 3. Backward computation in hidden layers
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
     * Backward computation to get gradients. Gradients are computed and saved in each layer based on backward errors.
     * 
     * @param predicts
     *            predicted values
     * @param actuals
     *            the actual target values
     * @param sig
     *            array of weights for targets
     * @return null as gradients are accumulated in backward computation of each layer.
     */
    @SuppressWarnings("rawtypes")
    public double[] backward(double[] predicts, double[] actuals, float[] sig) {
        // TODO add binary cross entropy here, merge into another backward method
        // 1. Error computation based on outputs of different target.
        double[] grad2Logits = new double[predicts.length];
        for(int i = 0; i < grad2Logits.length; i++) {
            double error = (predicts[i] - actuals[i]);
            grad2Logits[i] = error * (CommonUtils.sigmoidDerivedFunction(predicts[i]) + FLAT_SPOT_VALUE) * sig[i] * -1d;
        }

        // 2. Final layers backward accumulation
        double[] backInputs = new double[finalLayers.get(0).getIn()];
        for(int i = 0; i < this.finalLayers.size(); i++) {
            backInputs = CommonUtils.plus(backInputs,
                    this.finalLayers.get(i).backward(new double[] { grad2Logits[i] }));
        }

        // 3. Backward computation in hidden layers
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
     * Each layer with correlated optimizer should be optimized for weights updating.
     */
    @SuppressWarnings("rawtypes")
    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        for(DenseLayer finalLayer: this.finalLayers) {
            finalLayer.initOptimizer(learningRate, algorithm, reg, rl);
        }
        for(Layer layer: this.hiddenLayers) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initOptimizer(learningRate, algorithm, reg, rl);
            }
        }
    }

    /**
     * Update model weights based on given gradients per each layer.
     */
    @SuppressWarnings("rawtypes")
    @Override
    public void optimizeWeight(double numTrainSize, int iteration, MultiTaskModel gndModel) {
        // no need update input layer.
        List<Layer> gradHLs = gndModel.hiddenLayers;
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if(tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).optimizeWeight(numTrainSize, iteration, (DenseLayer) gradHLs.get(i));
            }
        }
        for(int i = 0; i < this.finalLayers.size(); i++) {
            this.finalLayers.get(i).optimizeWeight(numTrainSize, iteration, gndModel.getFinalLayers().get(i));
        }
    }

    /**
     * Combine gradients for all layers. This is used in master gradients aggregation to aggregate all gradients from
     * different workers.
     */
    @SuppressWarnings("rawtypes")
    @Override
    public MultiTaskModel combine(MultiTaskModel from) {
        this.dil.combine(from.getDil());

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

        for(int i = 0; i < this.finalLayers.size(); i++) {
            this.finalLayers.get(i).combine(from.finalLayers.get(i));
        }
        return this;
    }

    /**
     * Serialization based on different serialization type. This is used for master worker message passing and model
     * spec saving as a model spec file.
     */
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

        out.writeInt(this.finalLayers.size());
        for(int i = 0; i < this.finalLayers.size(); i++) {
            writeLayerWithNuLLCheck(out, this.finalLayers.get(i));
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
            SerializationUtil.writeIntList(out, hiddenNodes);
            out.writeDouble(l2reg);
            SerializationUtil.writeIntList(out, finalOutputs);
        }
    }

    /**
     * Write abstract layer with 'null' check.
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
     * Read layer with 'null' check.
     * 
     * @param in
     *            the input stream to be read
     * @param layer
     *            the layer to hold serialized data. This value should not be null.
     * @return de-serialized layer instance
     * @throws IOException
     *             if any exception in reading to a layer instance.
     */
    @SuppressWarnings("rawtypes")
    private AbstractLayer readLayerWithNullCheck(DataInput in, AbstractLayer layer) throws IOException {
        if(in.readBoolean()) {
            layer.readFields(in, this.serializationType);
        }
        return layer;
    }

    /**
     * De-serialize one multiple task model based on input stream which is reverse of {@link #write(DataOutput)}.
     * This is used to read model from file to instance or message transfer to master or workers.
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        this.serializationType = SerializationType.getSerializationType(in.readInt());

        this.dil = (DenseInputLayer) readLayerWithNullCheck(in, new DenseInputLayer());

        List<DenseLayer> hiddenDenseLayer = new ArrayList<>();
        int size = in.readInt();
        for(int i = 0; i < size; i++) {
            DenseLayer denseLayer = new DenseLayer();
            denseLayer.readFields(in, this.serializationType);
            hiddenDenseLayer.add(denseLayer);
        }

        int finalSize = in.readInt();
        this.finalLayers = new ArrayList<>(finalSize);
        for(int i = 0; i < finalSize; i++) {
            this.finalLayers.add((DenseLayer) readLayerWithNullCheck(in, new DenseLayer()));
        }

        this.actiFuncs = new ArrayList<>();
        size = in.readInt();
        for(int i = 0; i < size; i++) {
            this.actiFuncs.add(in.readUTF());
        }

        AssertUtils.assertListNotNullAndSizeEqual(this.actiFuncs, hiddenDenseLayer);
        this.hiddenLayers = new ArrayList<>(this.actiFuncs.size() * 2);
        for(int i = 0; i < hiddenDenseLayer.size(); i++) {
            this.hiddenLayers.add(hiddenDenseLayer.get(i));
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(this.actiFuncs.get(i)));
        }

        if(serializationType == SerializationType.MODEL_SPEC) {
            hiddenNodes = SerializationUtil.readIntList(in, hiddenNodes);
            l2reg = in.readDouble();
            hiddenNodes = SerializationUtil.readIntList(in, finalOutputs);
        }
    }

    /**
     * Initialize model weights based on different {@link InitMethod}. Default is random in range [-1, 1] which is
     * consistent with Shifu NN default initializer.
     */
    @SuppressWarnings("rawtypes")
    @Override
    public void initWeight(InitMethod method) {
        for(Layer layer: this.hiddenLayers) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight(method);
            }
        }
        for(int i = 0; i < this.finalLayers.size(); i++) {
            this.finalLayers.get(i).initWeight(method);
        }
    }

    /**
     * Default weights initialization.
     */
    public void initWeights() {
        InitMethod defaultMode = InitMethod.NEGATIVE_POSITIVE_ONE_RANGE_RANDOM;
        initWeight(defaultMode);
        LOG.info("Init weight be called with mode:{}", defaultMode.name());
    }

    @SuppressWarnings("unused")
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

    /**
     * Update weights from existing multiple task model. This is mostly to be used in model training starting from
     * existing model or fail over from checkpoint model weights.
     */
    @Override
    public void initWeight(MultiTaskModel updateModel) {
        AssertUtils.assertListNotNullAndSizeEqual(this.hiddenLayers, updateModel.getHiddenLayers());
        for(int i = 0; i < this.hiddenLayers.size(); i++) {
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if(this.hiddenLayers.get(i) instanceof DenseLayer) {
                ((DenseLayer) this.hiddenLayers.get(i)).initWeight((DenseLayer) updateModel.getHiddenLayers().get(i));
            }
        }
        for(int i = 0; i < this.finalLayers.size(); i++) {
            this.finalLayers.get(i).initWeight(updateModel.getFinalLayers().get(i));
        }
    }

    public void write(DataOutput out, SerializationType serializationType) throws IOException {
        this.serializationType = serializationType;
        write(out);
    }

    @Override
    public MultiTaskModel clone() {
        // Set the initial buffer size to 1M
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(1024 * 1024);
        DataOutputStream dos = new DataOutputStream(byteArrayOutputStream);
        DataInputStream dis = null;
        try {
            write(dos, SerializationType.MODEL_SPEC);
            dos.flush();
            ByteArrayInputStream dataInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
            MultiTaskModel mtl = new MultiTaskModel();
            dis = new DataInputStream(dataInputStream);
            mtl.readFields(dis);
            mtl.initGrads();
            return mtl;
        } catch (IOException e) {
            LOG.error("IOException happen when clone MTL model", e);
        } finally {
            IOUtils.closeStream(dos);
            if(dis != null) {
                IOUtils.closeStream(dis);
            }
        }
        return null;
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
     * @param finalLayers
     *            the finalLayers to set
     */
    public void setFinalLayers(List<DenseLayer> finalLayers) {
        this.finalLayers = finalLayers;
    }

    /**
     * @return the finalLayers
     */
    public List<DenseLayer> getFinalLayers() {
        return finalLayers;
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
     * @return the finalOutputs
     */
    public List<Integer> getFinalOutputs() {
        return finalOutputs;
    }

    /**
     * @param finalOutputs
     *            the finalOutputs to set
     */
    public void setFinalOutputs(List<Integer> finalOutputs) {
        this.finalOutputs = finalOutputs;
    }

}
