package ml.shifu.plugin.encog.adapter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ml.shifu.core.plugin.pmml.GenericMLModelBuilder;

import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.NeuralInput;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.Neuron;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

public class EncogNeuralNetworkBuilder implements
        GenericMLModelBuilder<BasicNetwork, NeuralNetwork> {
    BasicNetwork mlModel = new BasicNetwork();
    private NeuralNetwork pmmlModel;
    private List<HashMap<String, Integer>> neuronMap = new ArrayList<HashMap<String, Integer>>();
    @SuppressWarnings("serial")
    HashMap<ActivationFunctionType, ActivationFunction> functionMap = new HashMap<ActivationFunctionType, ActivationFunction>() {
        {
            put(ActivationFunctionType.LOGISTIC, new ActivationSigmoid());
            put(ActivationFunctionType.IDENTITY, new ActivationLinear());
            put(ActivationFunctionType.TANH, new ActivationTANH());
        }
    };

    @Override
    public BasicNetwork createMLModelFromPMML(NeuralNetwork pmmlModel) {
        this.pmmlModel = pmmlModel;
        readNeuronInputLayer();
        initNNLayer();
        setWeight();
        return mlModel;
    }

    private ActivationFunction transformActivationFunction(
            ActivationFunctionType pmmlActivationFuncType) {
        return functionMap.get(pmmlActivationFuncType);
    }

    private void initNNLayer() {
        List<NeuralLayer> layerList = pmmlModel.getNeuralLayers();
        for (NeuralLayer layer : layerList) {
            mlModel.addLayer(new BasicLayer(transformActivationFunction(layer
                    .getActivationFunction()), true, layer.getNeurons().size()));
        }
        mlModel.getStructure().finalizeStructure();
    }

    private void setWeight() {
        List<NeuralLayer> layerList = pmmlModel.getNeuralLayers();
        HashMap<String, Integer> prevNameLocMap;
        int lenLayer = layerList.size();
        // get each layer
        for (int layerID = 0; layerID < lenLayer; layerID++) {
            NeuralLayer layer = layerList.get(layerID);
            prevNameLocMap = neuronMap.get(layerID);
            // create new nameLocMap
            HashMap<String, Integer> nameLocMap = new HashMap<String, Integer>();
            int neuronNum = layer.getNeurons().size();
 
            for (int nID = 0; nID < neuronNum; nID++) {
                Neuron neuron = layer.getNeurons().get(nID);
                // add to nameLocMap
                nameLocMap.put(neuron.getId(), nID);
                for (Connection con : neuron.getConnections()) {
                    if (!prevNameLocMap.containsKey(con.getFrom()))
                        System.out.println(con.getFrom());
                    mlModel.setWeight(layerID,
                            prevNameLocMap.get(con.getFrom()), nID,
                            con.getWeight());
                }
            }// end of each neuron
            nameLocMap.put("bias", neuronNum);
            neuronMap.add(nameLocMap);
        }// end of a neural layer
    }

    private void readNeuronInputLayer() {
        // get input
        HashMap<String, Integer> nameLocMap = new HashMap<String, Integer>();
        List<NeuralInput> inputs = pmmlModel.getNeuralInputs()
                .getNeuralInputs();
        for (int i = 0; i < inputs.size(); i++) {
            nameLocMap.put(inputs.get(i).getId(), i);
        }
        mlModel.addLayer(new BasicLayer(new ActivationLinear(), true, inputs
                .size()));
        neuronMap.add(nameLocMap);
    }
}
