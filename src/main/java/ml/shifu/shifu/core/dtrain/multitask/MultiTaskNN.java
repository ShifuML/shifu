package ml.shifu.shifu.core.dtrain.multitask;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MultiTaskNN implements WeightInitializer<MultiTaskNN>, Bytable, Combinable<MultiTaskNN>, PropOptimizer<MultiTaskNN> {

    private static final Logger LOG = LoggerFactory.getLogger(MultiTaskNN.class);

    private DenseInputLayer dil;

    private int inputSize;

    private List<Layer> hiddenLayers;

    private List<Integer> hiddenNodes;

    private List<String> HiddenActiFuncs;

    private DenseLayer finalLayer;

    private int taskNumber;

    //    private List<String> finalActiFuncs;
    private Activation finalActiFunc;

    private double l2reg;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    public MultiTaskNN() {
    }

    public MultiTaskNN(int inputSize, List<Integer> hiddenNodes, List<String> HiddenActiFuncs, int taskNumber, double l2reg) {
        this.inputSize = inputSize;
        this.hiddenNodes = hiddenNodes;
        this.HiddenActiFuncs = HiddenActiFuncs;
        this.taskNumber = taskNumber;
//        this.finalActiFuncs = finalActiFuncs;
        this.l2reg = l2reg;

        // build the structure of NN graph.
        this.dil = new DenseInputLayer(this.inputSize);
        int preHiddenInputs = dil.getOutDim();

        AssertUtils.assertListNotNullAndSizeEqual(hiddenNodes, HiddenActiFuncs);
        this.hiddenLayers = new ArrayList<>(hiddenNodes.size() * 2);
        for (int i = 0; i < hiddenNodes.size(); i++) {
            int hiddenOutputs = hiddenNodes.get(i);
            DenseLayer denseLayer = new DenseLayer(hiddenOutputs, preHiddenInputs, l2reg);
            this.hiddenLayers.add(denseLayer);
            this.hiddenLayers.add(ActivationFactory.getInstance().getActivation(HiddenActiFuncs.get(i)));
            preHiddenInputs = hiddenOutputs;
        }

//        AssertUtils.assertEquals(taskNumber, finalActiFuncs.size());
        this.finalLayer = new DenseLayer(taskNumber, preHiddenInputs, l2reg);
        this.finalActiFunc = new Sigmoid();
    }

    public double[] forward(double[] denseInputs) {
        // input layer forward
        double[] dilOuts = this.dil.forward(denseInputs);
        double[] inputs = dilOuts;
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
        inputs = this.finalLayer.forward(inputs);
        //TODO:final Activation list.
        double[] resluts = this.finalActiFunc.forward(inputs);
        return resluts;
    }

    public float[] backward(double[] predicts, double[] actuals, double sig) {
        double[] grad2Logits = new double[predicts.length];

        for (int i = 0; i < grad2Logits.length; i++) {
            grad2Logits[i] = (predicts[i] - actuals[i]) * (predicts[i] * (1 - predicts[i])) * sig;
            // error * sigmoid derivertive * weight
        }

        double[] backInputs = this.finalLayer.backward(grad2Logits);
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
        return HiddenActiFuncs;
    }

    public void setHiddenActiFuncs(List<String> hiddenActiFuncs) {
        this.HiddenActiFuncs = hiddenActiFuncs;
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

    public void updateWeights(MultiTaskNN multiTask) {
        this.initWeight(multiTask);
    }

    public void updateWeights(MTNNParams params) {
        updateWeights(params.getMtnn());
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

//    // update model with optimizer
//    public void update(MultiTaskNN gradMultiTask, Optimizer optimizer) {
//        this.dil.update(gradMultiTask.getDil(), optimizer);
//
//        List<Layer> gradHLs = gradMultiTask.getHiddenLayers();
//        int hlSize = this.hiddenLayers.size();
//        for (int i = 0; i < hlSize; i++) {
//            Layer tmpLayer = this.hiddenLayers.get(i);
//            if (tmpLayer instanceof DenseLayer) {
//                ((DenseLayer) tmpLayer).update((DenseLayer) gradHLs.get(i), optimizer);
//            }
//        }
//
//        this.finalLayer.update(gradMultiTask.getFinalLayer(), optimizer);
//    }

    @Override
    public void write(DataOutput out) throws IOException {
        //todo
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        //todo
    }

    @Override
    public MultiTaskNN combine(MultiTaskNN from) {
        return null;
    }

    @Override
    public void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl) {
        for (Layer layer : this.hiddenLayers) {
            if (layer instanceof DenseLayer) {
                ((DenseLayer) layer).initOptimizer(learningRate, algorithm, reg, rl);
            }
        }
        this.finalLayer.initOptimizer(learningRate, algorithm, reg, rl);
    }

    @Override
    public void optimizeWeight(double numTrainSize, int iteration, MultiTaskNN model) {
        List<Layer> gradHLs = model.getHiddenLayers();
        for (int i = 0; i < this.hiddenLayers.size(); i++) {
            Layer tmpLayer = this.hiddenLayers.get(i);
            if (tmpLayer instanceof DenseLayer) {
                ((DenseLayer) tmpLayer).optimizeWeight(numTrainSize, iteration, (DenseLayer) gradHLs.get(i));
            }
        }
        this.finalLayer.optimizeWeight(numTrainSize, iteration, model.getFinalLayer());
    }
}
