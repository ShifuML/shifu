package ml.shifu.plugin.encog.adapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import ml.shifu.plugin.PMMLAdapterCommonUtil;
import ml.shifu.plugin.PMMLModelBuilder;

import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.Neuron;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.neural.flat.FlatNetwork;

/**
 * The class that converts an Encog NeuralNetwork to a PMML RegressionModel.
 * This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,Encog.NeuralNetwork>.
 * 
 */
public class PMMLEncogNeuralNetworkModel
        implements
        PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, org.encog.neural.networks.BasicNetwork> {

    private FlatNetwork network;

    /**
     * <p>
     * The function which converts an Encog NeuralNetwork to a PMML
     * NeuralNetwork Model.
     * <p>
     * This function reads the weights from the Encog NeuralNetwork model and
     * assign them to the corresponding connections of Neurons in PMML model.
     * 
     * @param bNetwork
     *            Encog NeuralNetwork
     * @return The generated PMML NeuralNetwork Model
     */
    public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(
            org.encog.neural.networks.BasicNetwork bNetwork,
            org.dmg.pmml.NeuralNetwork pmmlModel) {
        network = bNetwork.getFlat();
        MiningSchema schema = pmmlModel.getMiningSchema();
        pmmlModel.withNeuralInputs(PMMLAdapterCommonUtil
                .getNeuralInputs(schema));
        pmmlModel
                .withLocalTransformations(PMMLAdapterCommonUtil
                        .getBiasLocalTransformation(pmmlModel
                                .getLocalTransformations()));

        int[] layerCount = network.getLayerCounts();
        int[] layerFeedCount = network.getLayerFeedCounts();
        double[] weights = network.getWeights();
        ActivationFunctionType[] functionList = transformActivationFunction(network
                .getActivationFunctions());

        int numLayers = layerCount.length;
        int weightID = 0;
        List<NeuralLayer> layerList = new ArrayList<NeuralLayer>();
        String biasValue = "bias";
        pmmlModel.withFunctionName(MiningFunctionType.REGRESSION);
        for (int i = 0; i < numLayers - 1; i++) {
            NeuralLayer layer = new NeuralLayer();
            layer.setNumberOfNeurons(layerFeedCount[i]);
            layer.setActivationFunction(functionList[i]);
            int layerID = numLayers - i - 1;
            for (int j = 0; j < layerFeedCount[i]; j++) {
                Neuron neuron = new Neuron(String.valueOf(layerID + "," + j));
                neuron.setBias(0.0);// bias of each neuron, set to 0

                for (int k = 0; k < layerFeedCount[i + 1]; k++) {
                    neuron.withConnections(new Connection(String
                            .valueOf(layerID - 1 + "," + k),
                            weights[weightID++]));
                }// weights
                int tmp = layerCount[i + 1] - layerFeedCount[i + 1];
                // TODO set bias as constant, don't need to read from field
                for (int k = 0; k < tmp; k++) {
                    neuron.withConnections(new Connection(biasValue,
                            weights[weightID++]));
                }// bias neuron for each layer, set to bias=1
                layer.withNeurons(neuron);
            }// finish build Neuron
            layerList.add(layer);
        }// finish build layer
         // reserve the layer list to fit fot PMML format
        Collections.reverse(layerList);
        pmmlModel.withNeuralLayers(layerList);
        // set neural output based on target id
        pmmlModel.withNeuralOutputs(PMMLAdapterCommonUtil.getOutputFields(schema, numLayers - 1));
        deleteTargetDerivedFields(pmmlModel);
        return pmmlModel;
    }

    private ActivationFunctionType[] transformActivationFunction(
            ActivationFunction[] functions) {
        int funLen = functions.length;
        ActivationFunctionType[] functionType = new ActivationFunctionType[funLen];
        @SuppressWarnings("serial")
        HashMap<String, ActivationFunctionType> functionMap = new HashMap<String, ActivationFunctionType>() {
            {
                put("ActivationSigmoid", ActivationFunctionType.LOGISTIC);
                put("ActivationLinear", ActivationFunctionType.IDENTITY);
                put("ActivationTANH", ActivationFunctionType.TANH);
            }
        };
        for (int i = 0; i < funLen; i++) {
            String trimS = functions[i].getClass().getName();
            String[] functionS = trimS.split("\\.");
            functionType[i] = functionMap.get(functionS[functionS.length - 1]);
        }
        return functionType;
    }
private org.dmg.pmml.NeuralNetwork deleteTargetDerivedFields(org.dmg.pmml.NeuralNetwork pmmlModel) {
    pmmlModel.getLocalTransformations().getDerivedFields().remove(0);
    
    return pmmlModel;
}

}
