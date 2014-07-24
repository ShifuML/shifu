package ml.shifu.plugin.mahout.adapter;

import ml.shifu.core.plugin.pmml.AdapterConstants;
import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelBuilder;

import org.apache.mahout.classifier.mlp.NeuralNetwork;
import org.apache.mahout.math.Matrix;
import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.Neuron;


public class PMMLMahoutNeuralNetworkModel implements
        PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, NeuralNetwork> {
  
    public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(NeuralNetwork nnModel,
            org.dmg.pmml.NeuralNetwork pmmlModel) {
        Matrix[] matrixList = nnModel.getWeightMatrices();
        MiningSchema schema = pmmlModel.getMiningSchema();
        pmmlModel.withFunctionName(MiningFunctionType.REGRESSION);
        for (int layerIndex = 0; layerIndex < matrixList.length; layerIndex++) {
            NeuralLayer layer = new NeuralLayer();
            Matrix matrix = matrixList[layerIndex];
            int rowSize = matrix.numRows();
            int columnSize = matrix.numCols();
            layer.setNumberOfNeurons(rowSize);
            // TODO since squashFunctionList in Mahout is not accessible, by
            // default, we set ActivationFunction to sigmoid
            layer.setActivationFunction(ActivationFunctionType.LOGISTIC);
            for (int j = 0; j < rowSize; j++) {
                Neuron neuron = new Neuron(String.valueOf((layerIndex + 1)
                        + "," + j));
                neuron.setBias(0.0);// bias of each neuron, set to 0

                for (int k = 1; k < columnSize; k++) {
                    neuron.withConnections(new Connection(String
                            .valueOf(layerIndex + "," + (k - 1)), matrix.get(j,
                            k)));
                }
                // bias neuron for each layer, set to bias=1
                neuron.withConnections(new Connection(AdapterConstants.biasValue, matrix.get(j,
                        0)));
                layer.withNeurons(neuron);
            }// finish build Neuron
            pmmlModel.withNeuralLayers(layer);
        }// finish build layer
        // TODO outputID: classify to M or B based on the input
        pmmlModel.withNeuralOutputs(PMMLAdapterCommonUtil.getOutputFields(schema, matrixList.length));
        return pmmlModel;
    }

    // private ActivationFunctionType[] transformActivationFunction(
    // ActivationFunction[] functions) {
    // int funLen = functions.length;
    // ActivationFunctionType[] functionType = new
    // ActivationFunctionType[funLen];
    // HashMap<String, ActivationFunctionType> functionMap = new HashMap<String,
    // ActivationFunctionType>() {
    // {
    // put("Sigmoid", ActivationFunctionType.LOGISTIC);
    // put("Identity", ActivationFunctionType.IDENTITY);
    // }
    // };
    // for (int i = 0; i < funLen; i++) {
    // String trimS = functions[i].getClass().getName();
    // String[] functionS = trimS.split("\\.");
    // functionType[i] = functionMap.get(functionS[functionS.length - 1]);
    // }
    // return functionType;
    // }

}
