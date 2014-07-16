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

import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelBuilder;

/**
 * The class that converts the Spark LogisticRegressionModel to a PMML
 * RegressionModel. This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,spark.LogisticRegressionModel>.
 * 
 */
public class PMMLSparkLogisticRegressionModel
		implements
		PMMLModelBuilder<org.dmg.pmml.RegressionModel, org.apache.spark.mllib.classification.LogisticRegressionModel> {

	/**
	 * The function which converts the Spark LogisticRegressionModel to a PMML
	 * RegressionModel.
	 * 
	 * @param lrModel
	 *            Spark LogisticRegressionModel
	 * @param utility
	 *            DataFieldUtility that provides supplementary data field for
	 *            the model conversion
	 * 
	 * @return The generated PMML RegressionModel
	 */
	public org.dmg.pmml.RegressionModel adaptMLModelToPMML(
			org.apache.spark.mllib.classification.LogisticRegressionModel lrModel,
			org.dmg.pmml.RegressionModel pmmlModel) {
		double[] weights = lrModel.weights().toArray();
		double intercept = weights[0];// lrModel.intercept();

		return PMMLAdapterCommonUtil.getRegressionTable(weights, intercept,
				pmmlModel);
	}

}
