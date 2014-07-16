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

import java.util.HashMap;
import java.util.List;

import ml.shifu.core.plugin.pmml.GenericMLModelBuilder;

import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

public class EncogLogisticRegressionBuilder implements
		GenericMLModelBuilder<BasicNetwork, RegressionModel> {
	BasicNetwork mlModel = new BasicNetwork();
	private RegressionModel pmmlModel;
	@SuppressWarnings("serial")
	HashMap<RegressionNormalizationMethodType, ActivationFunction> functionMap = new HashMap<RegressionNormalizationMethodType, ActivationFunction>() {
		{
			put(RegressionNormalizationMethodType.LOGIT,
					new ActivationSigmoid());
			put(RegressionNormalizationMethodType.SOFTMAX,
					new ActivationSigmoid());
			put(RegressionNormalizationMethodType.NONE, new ActivationLinear());
		}
	};

	@Override
	public BasicNetwork createMLModelFromPMML(RegressionModel pmmlModel) {
		this.pmmlModel = pmmlModel;
		initNNLayer();
		setWeight();

		return mlModel;
	}

	private ActivationFunction transformActivationFunction(
			RegressionNormalizationMethodType pmmlActivationFuncType) {
		return functionMap.get(pmmlActivationFuncType);
	}

	private void initNNLayer() {
		mlModel.addLayer(new BasicLayer(new ActivationLinear(), true, pmmlModel
				.getRegressionTables().get(0).getNumericPredictors().size()));
		mlModel.addLayer(new BasicLayer(transformActivationFunction(pmmlModel
				.getNormalizationMethod()), false, 1));
		mlModel.getStructure().finalizeStructure();
	}

	private void setWeight() {
		List<NumericPredictor> nPredictors = pmmlModel.getRegressionTables()
				.get(0).getNumericPredictors();
		for (int i = 0; i < nPredictors.size(); i++) {
			mlModel.setWeight(0, i, 0, nPredictors.get(i).getCoefficient());
		}
	}

}
