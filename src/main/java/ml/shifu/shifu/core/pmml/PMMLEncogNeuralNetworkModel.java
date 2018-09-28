/**
 * Copyright [2012-2014] PayPal Software Foundation
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

package ml.shifu.shifu.core.pmml;

import org.dmg.pmml.neural_network.Connection;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork.ActivationFunction;
import org.dmg.pmml.neural_network.Neuron;
import org.encog.neural.flat.FlatNetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

/**
 * The class that converts an Encog NeuralNetwork to a PMML RegressionModel.
 * This class extends the abstract class
 * PMMLModelBuilder(pmml.RegressionModel,Encog.NeuralNetwork).
 */
public class PMMLEncogNeuralNetworkModel implements
        PMMLModelBuilder<org.dmg.pmml.neural_network.NeuralNetwork, org.encog.neural.networks.BasicNetwork> {

    private FlatNetwork network;

    /**
     * The function which converts an Encog NeuralNetwork to a PMML
     * NeuralNetwork Model.
     * <p>
     * This function reads the weights from the Encog NeuralNetwork model and assign them to the corresponding
     * connections of Neurons in PMML model.
     * 
     * @param bNetwork
     *            Encog NeuralNetwork
     * @param pmmlModel
     *            DataFieldUtility that provides supplementary data field for
     *            the model conversion
     * @return The generated PMML NeuralNetwork Model
     */
    public org.dmg.pmml.neural_network.NeuralNetwork adaptMLModelToPMML(org.encog.neural.networks.BasicNetwork bNetwork,
            org.dmg.pmml.neural_network.NeuralNetwork pmmlModel) {
        network = bNetwork.getFlat();
        pmmlModel = new NeuralNetworkModelIntegrator().adaptPMML(pmmlModel);
        int[] layerCount = network.getLayerCounts();
        int[] layerFeedCount = network.getLayerFeedCounts();
        double[] weights = network.getWeights();
        ActivationFunction[] functionList = transformActivationFunction(network.getActivationFunctions());

        int numLayers = layerCount.length;
        int weightID = 0;
        List<NeuralLayer> layerList = new ArrayList<NeuralLayer>();

        // by zhanhu - set the function name in NNPmmlModelCreator
        // pmmlModel.withFunctionName(MiningFunctionType.REGRESSION);
        for(int i = 0; i < numLayers - 1; i++) {
            NeuralLayer layer = new NeuralLayer();
            layer.setNumberOfNeurons(layerFeedCount[i]);
            layer.setActivationFunction(functionList[i]);
            int layerID = numLayers - i - 1;
            for(int j = 0; j < layerFeedCount[i]; j++) {
                Neuron neuron = new Neuron();
                neuron.setId(String.valueOf(layerID + "," + j));
                for(int k = 0; k < layerFeedCount[i + 1]; k++) {
                    neuron.addConnections(new Connection(String.valueOf(layerID - 1 + "," + k), weights[weightID++]));
                }// weights
                int tmp = layerCount[i + 1] - layerFeedCount[i + 1];
                for(int k = 0; k < tmp; k++) {
                    neuron.setBias(weights[weightID++]);
                } // bias neuron for each layer
                layer.addNeurons(neuron);
            }// finish build Neuron
            layerList.add(layer);
        }// finish build layer
         // reserve the layer list to fit fot PMML format
        Collections.reverse(layerList);
        pmmlModel.addNeuralLayers(layerList.toArray(new NeuralLayer[layerList.size()]));
        // set neural output based on target id
        pmmlModel.setNeuralOutputs(PMMLAdapterCommonUtil.getOutputFields(pmmlModel.getMiningSchema(), numLayers - 1));
        return pmmlModel;
    }

    private ActivationFunction[] transformActivationFunction(org.encog.engine.network.activation.ActivationFunction[] functions) {
        int funLen = functions.length;
        ActivationFunction[] functionType = new ActivationFunction[funLen];
        @SuppressWarnings("serial")
        HashMap<String, ActivationFunction> functionMap = new HashMap<String, ActivationFunction>() {
            {
                put("ActivationSigmoid", ActivationFunction.LOGISTIC);
                put("ActivationLinear", ActivationFunction.IDENTITY);
                put("ActivationTANH", ActivationFunction.TANH);
                put("ActivationReLU", ActivationFunction.RECTIFIER);
            }
        };
        for(int i = 0; i < funLen; i++) {
            String trimS = functions[i].getClass().getName();
            String[] functionS = trimS.split("\\.");
            functionType[i] = functionMap.get(functionS[functionS.length - 1]);
        }
        return functionType;
    }
}
