package ml.shifu.plugin.mahout.adapter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ml.shifu.plugin.GenericMLModelBuilder;

import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.mlp.NeuralNetwork;
import org.apache.mahout.math.Matrix;
import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.NeuralInput;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.Neuron;

public class MahoutNeuralNetworkBuilder implements
        GenericMLModelBuilder<NeuralNetwork, org.dmg.pmml.NeuralNetwork> {
    NeuralNetwork mlModel;
    private org.dmg.pmml.NeuralNetwork pmmlModel;
    private List<HashMap<String, Integer>> neuronMap = new ArrayList<HashMap<String, Integer>>();
    @SuppressWarnings("serial")
    HashMap<ActivationFunctionType, String> functionMap = new HashMap<ActivationFunctionType, String>() {
        {
            put(ActivationFunctionType.LOGISTIC, "Sigmoid");
            put(ActivationFunctionType.IDENTITY, "Identity");
        }
    };

    @Override
    public NeuralNetwork createMLModelFromPMML(
            org.dmg.pmml.NeuralNetwork pmmlModel) {
        this.pmmlModel = pmmlModel;
        initNNLayer();
        setWeight();
        return mlModel;
    }

    private void initNNLayer() {
        mlModel = new MultilayerPerceptron();
        HashMap<String, Integer> nameLocMap = new HashMap<String, Integer>();
        // input layer
        List<NeuralInput> inputs = pmmlModel.getNeuralInputs()
                .getNeuralInputs();
        int lenInput = inputs.size();
        for (int i = 0; i < lenInput-1; i++) {
            nameLocMap.put(inputs.get(i).getId(), i+1);
        }

        mlModel.addLayer(lenInput-1, false, "Identity");
        neuronMap.add(nameLocMap);
        // neural layers
        List<NeuralLayer> layerList = pmmlModel.getNeuralLayers();
        int lenLayer = layerList.size();
        for (int i = 0; i < lenLayer - 1; i++) {
            NeuralLayer layer = layerList.get(i);
            mlModel.addLayer(layer.getNeurons().size(), false,
                    functionMap.get(layer.getActivationFunction()));
            
        }
        NeuralLayer outputLayer = layerList.get(lenLayer - 1);
        mlModel.addLayer(outputLayer.getNeurons().size(),true,
                functionMap.get(outputLayer.getActivationFunction()));
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
            // get matrix
            Matrix matrix = mlModel.getWeightsByLayer(layerID);
            for (int nID = 0; nID < neuronNum; nID++) {
                Neuron neuron = layer.getNeurons().get(nID);
                // add to nameLocMap
                nameLocMap.put(neuron.getId(), nID+1);
                List<Connection> conList = neuron.getConnections();
                int conSize = conList.size();
                Connection biasCon = conList.get(conSize-1);
                matrix.set( nID,0,biasCon.getWeight());
                for (int i=0;i<conSize-1;i++) {
                    Connection con = conList.get(i);
                    matrix.set( nID,prevNameLocMap.get(con.getFrom()),
                            con.getWeight());
                }
            }// end of each neuron
            neuronMap.add(nameLocMap);
            mlModel.setWeightMatrix(layerID, matrix);
        }// end of a neural layer
    }

}
