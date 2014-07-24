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

package ml.shifu.plugin.encog.adapter;

import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelBuilder;

import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;

/**
 * The class that converts a special Encog NeuralNetwork without hidden layers
 * to a PMML RegressionModel. This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,Encog.NeuralNetwork>.
 * 
 */
public class PMMLEncogLogisticRegressionModel implements
		PMMLModelBuilder<org.dmg.pmml.RegressionModel, BasicNetwork> {
	private FlatNetwork network;

	/**
	 * The function which converts a special Encog NeuralNetwork without hidden
	 * layers, to a PMML RegressionModel.
	 * 
	 * @param bNetwork
	 *            Encog NeuralNetwork
	 * @param utility
	 *            DataFieldUtility that provides supplementary data field for
	 *            the model conversion
	 * @return The generated PMML RegressionModel
	 */
	public org.dmg.pmml.RegressionModel adaptMLModelToPMML(
			BasicNetwork bNetwork, org.dmg.pmml.RegressionModel pmmlModel) {
		network = bNetwork.getFlat();
		double[] weights = network.getWeights();
		return PMMLAdapterCommonUtil.getRegressionTable(weights, 0, pmmlModel);
	}

}
