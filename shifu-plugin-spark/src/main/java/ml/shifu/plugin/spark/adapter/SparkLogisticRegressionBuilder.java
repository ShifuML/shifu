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

package ml.shifu.plugin.spark.adapter;

import java.util.List;

import ml.shifu.core.plugin.pmml.GenericMLModelBuilder;

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionTable;

public class SparkLogisticRegressionBuilder implements
		GenericMLModelBuilder<LogisticRegressionModel, RegressionModel> {
	LogisticRegressionModel mlModel;

	@Override
	public LogisticRegressionModel createMLModelFromPMML(
			RegressionModel pmmlModel) {

		RegressionTable rTable = pmmlModel.getRegressionTables().get(0);
		List<NumericPredictor> nPredictors = rTable.getNumericPredictors();
		double intercept = rTable.getIntercept();
		double[] coefficients = new double[nPredictors.size() + 1];
		coefficients[0] = intercept;
		for (int i = 0; i < nPredictors.size(); i++) {
			coefficients[i + 1] = nPredictors.get(i).getCoefficient();
		}
		Vector vector = new DenseVector(coefficients);
		mlModel = new LogisticRegressionModel(vector, intercept);
		mlModel.clearThreshold();
		return mlModel;
	}

}
