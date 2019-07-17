package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.core.dtrain.wdl.*;

import ml.shifu.shifu.core.dtrain.wdl.activation.Activation;
import ml.shifu.shifu.core.dtrain.wdl.activation.ActivationFactory;
import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MultiTaskNN implements WeightInitializer<MultiTaskNN>, Bytable, Combinable<MultiTaskNN> {

    private static final Logger LOG = LoggerFactory.getLogger(MultiTaskNN.class);

    private DenseInputLayer dil;

    private List<Layer> hiddenLayers;

    private DenseLayer finalLayer;

    private int numericalSize;

    private List<Integer> hiddenNodes;

    private List<String> actiFuncs;

    private float l2reg;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    public MultiTaskNN() {
    }

    public MultiTaskNN(int numericalSize, List<Integer> hiddenNodes, List<String> actiFuncs, float l2reg) {
        this.numericalSize = numericalSize;
        this.hiddenNodes = hiddenNodes;
        this.actiFuncs = actiFuncs;
        this.l2reg = l2reg;

        // build the structure of NN graph.
        this.dil = new DenseInputLayer(numericalSize);
        int preHiddenInputs = dil.getOutDim();

        AssertUtils.assertListNotNullAndSizeEqual(hiddenNodes, actiFuncs);
        this.hiddenLayers = new ArrayList<>(hiddenNodes.size() * 2);
        for (int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs, l2reg);
            this.hiddenLayers.add(denseLayer);
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(actiFuncs.get(i)));
        }

        this.finalLayer = new DenseLayer(1, preHiddenInputs, l2reg);
    }

    public float[] forward(float[] denseInputs) {
        // input layer forward
        float[] dilOuts = this.dil.forward(denseInputs);
        float[] inputs = dilOuts;
        // hidden layers forward
        for (Layer layer : this.hiddenLayers) {
            if (layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                inputs = denseLayer.forward(inputs);
            } else if (layer instanceof Activation) {
                Activation acti = (Activation) layer;
                inputs = acti.forward(inputs);
            }
        }
        //final layer forward
        float[] logits = this.finalLayer.forward(inputs);
        return logits;
    }

    public float[] backward(float[] predicts, float[] actuals, float sig) {
        float[] grad2Logits = new float[predicts.length];

        for (int i = 0; i < grad2Logits.length; i++) {
            grad2Logits[i] = (predicts[i] - actuals[i]) * (predicts[i] * (1 - predicts[i])) * sig;
            // error * sigmoid derivertive * weight
        }

        float[] backInputs = this.finalLayer.backward(grad2Logits);
        for (int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1 - i);
            if (layer instanceof DenseLayer) {
                backInputs = ((DenseLayer) layer).backward(backInputs);
            } else if (layer instanceof Activation) {
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
        for (Layer layer : hiddenLayers) {
            if (layer instanceof DenseLayer) {
                ((DenseLayer) layer).initGrads();
            }
        }
        this.finalLayer.initGrads();
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


    public int getNumericalSize() {
        return numericalSize;
    }

    public void setNumericalSize(int numericalSize) {
        this.numericalSize = numericalSize;
    }

    public List<Integer> getHiddenNodes() {
        return hiddenNodes;
    }

    public void setHiddenNodes(List<Integer> hiddenNodes) {
        this.hiddenNodes = hiddenNodes;
    }

    public List<String> getActiFuncs() {
        return actiFuncs;
    }

    public void setActiFuncs(List<String> actiFuncs) {
        this.actiFuncs = actiFuncs;
    }

    public float getL2reg() {
        return l2reg;
    }

    public void setL2reg(float l2reg) {
        this.l2reg = l2reg;
    }

    public SerializationType getSerializationType() {
        return serializationType;
    }

    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    public void updateWeights(MultiTaskNN multiTask) {
        this.initWeight(multiTask);
    }

    public void updateWeights(MTNNParams params) {
        //todo:params
        //updateWeights(params);
        // after update weights, gradients should be re newed.
        this.initGrads();
    }

    /**
     * Init the weights in MultiTaskNN Model and it's sub module
     */
    public void initWeights() {
        InitMethod defaultMode = InitMethod.ZERO_ONE_RANGE_RANDOM;
        initWeight(defaultMode);
        LOG.error("Init weight be called with mode:" + defaultMode.name());
    }

    @Override
    public void initWeight(InitMethod method) {
        for (Layer layer : this.hiddenLayers) {
            if (layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight(method);
            }
        }
        this.finalLayer.initWeight(method);
    }

    @Override
    public void initWeight(MultiTaskNN updateModel) {
        AssertUtils.assertListNotNullAndSizeEqual(this.hiddenLayers, updateModel.getHiddenLayers());
        for (int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer layer = this.hiddenLayers.get(i);
            // There are two type of layer: DenseLayer, Activation. We only need to init DenseLayer
            if (layer instanceof DenseLayer) {
                ((DenseLayer) layer).initWeight((DenseLayer) updateModel.getHiddenLayers().get(i));
            }
            this.finalLayer.initWeight(updateModel.getFinalLayer());
        }
    }

    // update model with optimizer
    public void update(MultiTaskNN gradMultiTask, Optimizer optimizer) {
        this.dil.update(gradMultiTask.getDil(), optimizer);

        List<Layer> gradHLs = gradMultiTask.getHiddenLayers();
        int hlSize = this.hiddenLayers.size();
        for (int i = 0; i < hlSize; i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if (tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).update((DenseLayer) gradHLs.get(i), optimizer);
            }
        }

        this.finalLayer.update(gradMultiTask.getFinalLayer(), optimizer);
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    @Override
    public MultiTaskNN combine(MultiTaskNN from) {
        return null;
    }

}
