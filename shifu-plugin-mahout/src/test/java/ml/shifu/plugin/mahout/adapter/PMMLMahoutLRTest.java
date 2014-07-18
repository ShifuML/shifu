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
import java.util.Map;

import ml.shifu.core.util.PMMLUtils;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.DenseVector;
import org.dmg.pmml.FieldName;
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
 * Test PMMLEncogNeuralNetworkModel that converts an Encog NeuralNetwork model
 * to a PMML NeuralNetwork Model.
 */
public class PMMLMahoutLRTest {
	OnlineLogisticRegression lrModel;
	PMML pmml;
	private static Logger log = LoggerFactory.getLogger(PMMLMahoutLRTest.class);
	RegressionModelEvaluator evaluator;
	protected final double DELTA = Math.pow(10, -5);
	private String inputData = "src/test/resources/data/wdbc/inputTrainData";
	private String initPmmlPath = "src/test/resources/data/wdbc/model.xml";
	private String outputPMMLPath = "src/test/resources/adapter/mahoutLR/MahoutLR.pmml";
	private String evalFilePath = "src/test/resources/data/wdbc/evalData";

	protected void initMLModel() {
		try {
			pmml = PMMLUtils.loadPMML(initPmmlPath);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		List<MahoutDataPair> inputDataSet = new CommonUtil(inputData, pmml)
				.getMahoutDataPair();
		lrModel = new OnlineLogisticRegression(2, 30, new L1());
		for (MahoutDataPair pair : inputDataSet) {
			lrModel.train(pair.getActual(), pair.getVector());
		}

	}

	protected void adaptToPMML() {
		Model pmmlNN = pmml.getModels().get(0);
		RegressionModel pmmlLR = new RegressionModel()
				.withMiningSchema(pmmlNN.getMiningSchema())
				.withTargets(pmmlNN.getTargets())
				.withModelStats(pmmlNN.getModelStats())
				.withLocalTransformations(pmmlNN.getLocalTransformations());

		pmmlNN = new PMMLMahoutLogisticRegressionModel().adaptMLModelToPMML(
				lrModel, pmmlLR);
		pmml.getModels().set(0, pmmlLR);
	}

	protected void writeToPMML() {

		PMMLUtils.savePMML(pmml, outputPMMLPath);
		log.info(" - write PMML LogisticRegression model to " + outputPMMLPath
				+ "\n - the number of nodes in each layer ...");
	}

	protected void evaluatePMML() {
		evaluator = new RegressionModelEvaluator(pmml);

		CommonUtil evalInput = new CommonUtil(evalFilePath, pmml);
		evaluateInputs(evalInput);

	}

	private void evaluateInputs(CommonUtil evalInput) {
		log.info(" evaluate Mahout LR adapter with "
				+ evalInput.getEvaluatorInput().size() + " inputs");
		for (Map<FieldName, String> map : evalInput.getEvaluatorInput()) {
			ModelEvaluationContext context = new ModelEvaluationContext(null,
					evaluator);
			context.declareAll(map);
			double[] data = evalInput.normalizeData(context);
//			 System.out.println("pmml "+getPMMLEvaluatorResult(map)+" mlModel "+lrModel.classifyScalar(new DenseVector(data)));
			Assert.assertEquals(getPMMLEvaluatorResult(map),
					lrModel.classifyScalar(new DenseVector(data)), DELTA);
		}
	}

	private double getPMMLEvaluatorResult(Map<FieldName, String> inputData) {
		@SuppressWarnings("unchecked")
		Map<FieldName, Double> evalMap = (Map<FieldName, Double>) evaluator
				.evaluate(inputData);
		for (Map.Entry<FieldName, Double> entry : evalMap.entrySet()) {
			return entry.getValue();
		}
		return 0;
	}

	@Test
	public void testMahoutLR() {
		initMLModel();
		adaptToPMML();
		writeToPMML();
		evaluatePMML();
	}

}