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

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Map;

import ml.shifu.core.util.PMMLUtils;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.RegressionModel;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
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
public class PMMLEncogLRTest {
	BasicNetwork mlModel;
	PMML pmml;
	private static Logger log = LoggerFactory.getLogger(PMMLEncogLRTest.class);
	RegressionModelEvaluator evaluator;
	protected final double DELTA = Math.pow(10, -1)*2;
	private String mlModelPath = "src/test/resources/evaluator/encogLR/EncogLR.lr";
	private String initPmmlPath = "src/test/resources/evaluator/model.xml";
	private String outputPMMLPath = "src/test/resources/evaluator/encogLR/EncogLR_output.pmml";
	private String evalFilePath = "src/test/resources/evaluator/wdbc.train";

	protected void initMLModel() {
		try {
			pmml = PMMLUtils.loadPMML(initPmmlPath);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		PersistBasicNetwork networkReader = new PersistBasicNetwork();
		try {
			mlModel = (BasicNetwork) networkReader.read(new FileInputStream(
					mlModelPath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	protected void adaptToPMML() {
		Model pmmlNN = pmml.getModels().get(0);
		RegressionModel pmmlLR = new RegressionModel()
				.withMiningSchema(pmmlNN.getMiningSchema())
				.withTargets(pmmlNN.getTargets())
				.withModelStats(pmmlNN.getModelStats())
				.withLocalTransformations(pmmlNN.getLocalTransformations());

		pmmlNN = new PMMLEncogLogisticRegressionModel().adaptMLModelToPMML(
				mlModel, pmmlLR);
		pmml.getModels().set(0, pmmlLR);
	}

	protected void writeToPMML() {

		PMMLUtils.savePMML(pmml, outputPMMLPath);
		log.info(" - write PMML NeuralNetwork model to " + outputPMMLPath
				+ "\n - the number of nodes in each layer ...");
	}

	protected void evaluatePMML() {
		evaluator = new RegressionModelEvaluator(pmml);

		EvalCSVUtil evalInput = new EvalCSVUtil(evalFilePath, pmml);
		evaluateInputs(evalInput);

	}

	private void evaluateInputs(EvalCSVUtil evalInput) {
		log.info(" evaluate Encog LR adapter with " + evalInput.getEvaluatorInput().size()
				+ " inputs");
		 for (Map<FieldName, String> map : evalInput.getEvaluatorInput()) {
				 ModelEvaluationContext context = new ModelEvaluationContext(null,
				 evaluator);
				 context.declareAll(map);
				 MLData data = evalInput.normalizeData(context);
//				 System.out.println("pmml "+getPMMLEvaluatorResult(map)+" mlModel "+mlModel.compute(data).getData(0));
			Assert.assertEquals(
					getPMMLEvaluatorResult(map),
					mlModel.compute(data).getData(0), DELTA);
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
	public void testEncogLR() {
		initMLModel();
		adaptToPMML();
		writeToPMML();
		evaluatePMML();
	}
	
}