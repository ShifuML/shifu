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

import java.util.Map;

import ml.shifu.core.util.PMMLUtils;
import ml.shifu.plugin.spark.trainer.SparkCommonUtil;
import ml.shifu.plugin.spark.trainer.SparkUtility;
import ml.shifu.plugin.spark.trainer.StaticFunctions.ParsePoint;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.RegressionModel;
import org.jpmml.evaluator.ModelEvaluationContext;
import org.jpmml.evaluator.RegressionModelEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Test PMMLSparkLogisticRegressionModel that converts a Spark
 * LogisticRegressionModel to PMML Regression Model.
 * 
 */
public class PMMLSparkLogisticRegressionModelTest {
	org.apache.spark.mllib.classification.LogisticRegressionModel mlModel;
	PMML pmml;
	private String initPmmlPath = "src/test/resources/data/wdbc/model.xml";
	private String inputData = "src/test/resources/data/wdbc/inputTrainData";
	private String outputPMMLPath = "src/test/resources/adapter/sparkLR/SparkLR_output.pmml";
	private String evalFilePath = "src/test/resources/data/wdbc/evalData.txt";
	private static Logger log = LoggerFactory
			.getLogger(PMMLSparkLogisticRegressionModelTest.class);
	private RegressionModelEvaluator evaluator;
	protected final double DELTA = Math.pow(10, -5);

	protected void initMLModel() {
		try {
			pmml = PMMLUtils.loadPMML(initPmmlPath);
			Model model = pmml.getModels().get(0);
			double stepSize = 10.0;
			int iterations = 2;
			// training
			JavaRDD<String> distData = SparkUtility.getSc().textFile(inputData);
			ParsePoint parseFunc = new ParsePoint(SparkCommonUtil
					.getFieldIDViaUsageType(model.getMiningSchema(),
							FieldUsageType.TARGET).get(0),
					SparkCommonUtil.getActiveFields(model.getMiningSchema()),
					",");
			RDD<LabeledPoint> datas = distData.map(parseFunc).cache().rdd();
			mlModel = LogisticRegressionWithSGD.train(datas, iterations,
					stepSize).clearThreshold();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

	protected void adaptToPMML() {
		Model pmmlNN = pmml.getModels().get(0);
		RegressionModel pmmlLR = new RegressionModel()
				.withMiningSchema(pmmlNN.getMiningSchema())
				.withTargets(pmmlNN.getTargets())
				.withModelStats(pmmlNN.getModelStats())
				.withLocalTransformations(pmmlNN.getLocalTransformations());

		pmmlNN = new PMMLSparkLogisticRegressionModel().adaptMLModelToPMML(
				mlModel, pmmlLR);
		pmml.getModels().set(0, pmmlLR);
	}

	protected void writeToPMML() {
		PMMLUtils.savePMML(pmml, outputPMMLPath);
		log.info(" - write PMML Regression model to " + outputPMMLPath);
	}

	protected void evaluatePMML() {
		evaluator = new RegressionModelEvaluator(pmml);
		EvalCSVUtil evalInput = new EvalCSVUtil(evalFilePath, pmml);
		evaluateInputs(evalInput);

	}

	@Test
	public void testSparkLR() {
		initMLModel();
		adaptToPMML();
		writeToPMML();
		evaluatePMML();
	}

	private void evaluateInputs(EvalCSVUtil evalInput) {

		for (Map<FieldName, String> map : evalInput.getEvaluatorInput()) {
			ModelEvaluationContext context = new ModelEvaluationContext(null,
					evaluator);
			context.declareAll(map);
			Vector vector = new DenseVector(evalInput.normalizeData(context));
			// System.out.println("pmml "+getPMMLEvaluatorResult(map)+" mlModel "+
			// mlModel.predict(vector));
			Assert.assertEquals(getPMMLEvaluatorResult(map),
					mlModel.predict(vector), DELTA);
		}
	}

	protected double getPMMLEvaluatorResult(Map<FieldName, String> inputData) {
		if (evaluator == null)
			return 0;
		@SuppressWarnings("unchecked")
		Map<FieldName, Double> evalMap = (Map<FieldName, Double>) evaluator
				.evaluate(inputData);
		for (Map.Entry<FieldName, Double> entry : evalMap.entrySet()) {
			return entry.getValue();
		}
		return 0;
	}
}
