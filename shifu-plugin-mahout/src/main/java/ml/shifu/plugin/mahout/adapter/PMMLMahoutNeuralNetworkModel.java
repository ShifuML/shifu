/**
 * Copyright [2012-2014] eBay Software Foundation
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

package ml.shifu.plugin.mahout.adapter;

import ml.shifu.core.plugin.pmml.PluginConstants;
import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelBuilder;
import ml.shifu.core.plugin.pmml.NeuralNetworkModelIntegrator;

import org.apache.mahout.classifier.mlp.NeuralNetwork;
import org.apache.mahout.math.Matrix;
import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.Neuron;

public class PMMLMahoutNeuralNetworkModel implements
		PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, NeuralNetwork> {

	public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(NeuralNetwork nnModel,
			org.dmg.pmml.NeuralNetwork pmmlModel) {
		Matrix[] matrixList = nnModel.getWeightMatrices();
		pmmlModel.withFunctionName(MiningFunctionType.REGRESSION);
		pmmlModel = new NeuralNetworkModelIntegrator().adaptPMML(pmmlModel);
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
				neuron.withConnections(new Connection(
						PluginConstants.biasValue, matrix.get(j, 0)));
				layer.withNeurons(neuron);
			}// finish build Neuron
			pmmlModel.withNeuralLayers(layer);
		}// finish build layer
		pmmlModel.withNeuralOutputs(PMMLAdapterCommonUtil.getOutputFields(
				pmmlModel.getMiningSchema(), matrixList.length));
		return pmmlModel;
	}
}
