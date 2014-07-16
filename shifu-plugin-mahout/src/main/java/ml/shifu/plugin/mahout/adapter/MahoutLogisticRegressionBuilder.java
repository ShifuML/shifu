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

import java.util.List;

import ml.shifu.core.plugin.pmml.GenericMLModelBuilder;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;

public class MahoutLogisticRegressionBuilder implements
		GenericMLModelBuilder<OnlineLogisticRegression, RegressionModel> {
	OnlineLogisticRegression mlModel;
	private RegressionModel pmmlModel;

	@Override
	public OnlineLogisticRegression createMLModelFromPMML(
			RegressionModel pmmlModel) {
		this.pmmlModel = pmmlModel;
		initNNLayer();
		setWeight();
		return mlModel;
	}

	private void initNNLayer() {
		mlModel = new OnlineLogisticRegression(2, pmmlModel
				.getRegressionTables().get(0).getNumericPredictors().size(),
				new L1());
	}

	private void setWeight() {
		List<NumericPredictor> nPredictors = pmmlModel.getRegressionTables()
				.get(0).getNumericPredictors();
		for (int i = 0; i < nPredictors.size(); i++) {
			mlModel.setBeta(0, i, nPredictors.get(i).getCoefficient());
		}
	}

}
