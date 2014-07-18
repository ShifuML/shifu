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
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.PMML;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
import org.jpmml.evaluator.ModelEvaluationContext;
import org.jpmml.evaluator.NeuralNetworkEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Test PMMLEncogNeuralNetworkModel that converts an Encog NeuralNetwork model
 * to a PMML NeuralNetwork Model.
 */
public class PMMLEncogNeuralNetworkTest {
	BasicNetwork mlModel;
	PMML pmml;
	private static Logger log = LoggerFactory
			.getLogger(PMMLEncogNeuralNetworkTest.class);
	NeuralNetworkEvaluator evaluator;
	protected final double DELTA = Math.pow(10, -5);
	private String mlModelPath = "src/test/resources/adapter/encogNN/EncogNN.nn";
	private String initPmmlPath = "src/test/resources/data/wdbc/model.xml";
	private String outputPMMLPath = "src/test/resources/adapter/encogNN/EncogNN_output.pmml";
	private String evalFilePath = "src/test/resources/data/wdbc/evalData";

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
		NeuralNetwork pmmlNN = (NeuralNetwork) pmml.getModels().get(0);
		pmmlNN = new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML(mlModel,
				pmmlNN);
		pmml.getModels().set(0, pmmlNN);
	}

	protected void writeToPMML() {

		PMMLUtils.savePMML(pmml, outputPMMLPath);
		log.info(" - write PMML NeuralNetwork model to " + outputPMMLPath
				+ "\n - the number of nodes in each layer ...");
	}

	protected void evaluatePMML() {
		evaluator = new NeuralNetworkEvaluator(pmml);

		CommonUtil evalInput = new CommonUtil(evalFilePath, pmml);
		evaluateInputs(evalInput);

	}

	@Test
	public void testEncogNN_2layer() {
		initMLModel();
		adaptToPMML();
		writeToPMML();
		evaluatePMML();
	}

	private void evaluateInputs(CommonUtil evalInput) {
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


	

	//
	// private void evaluateNormalizedData() throws Exception {
	//
	// PMML pmmlNoStats = PMMLUtils.loadPMML(outputPMMLPath);
	// NeuralNetwork pmmlNN = (NeuralNetwork) pmmlNoStats.getModels().get(0);
	// List<String> activeFields = PMMLAdapterCommonUtil
	// .getSchemaActiveFields(pmmlNN.getMiningSchema());
	// DataDictionary dictionary = new DataDictionary();
	// for (String field : activeFields) {
	// DataField targetField = new DataField(new FieldName(field),
	// OpType.CONTINUOUS, DataType.DOUBLE);
	// // targetField.withValues(new Value("1")).withValues(new
	// // Value("0"));
	// dictionary.withDataFields(targetField);
	// }
	// pmmlNoStats.setDataDictionary(dictionary);
	// pmmlNN.setModelStats(null);
	//
	// DerivedField field = new DerivedField(OpType.CONTINUOUS,
	// DataType.DOUBLE).withName(new FieldName(
	// AdapterConstants.biasValue));
	// // field.withName(new FieldName(s));
	// field.withExpression(new
	// Constant(String.valueOf(AdapterConstants.bias)));
	// pmmlNN.setLocalTransformations(new LocalTransformations()
	// .withDerivedFields(field));
	// pmmlNoStats.withModels(pmmlNN);
	// // rebuild data dictionary
	// // copy mining schema
	// PMMLUtils.savePMML(pmmlNoStats,
	// "src/test/resources/encog/nn/EncogNN_noStats.pmml");
	// NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(
	// pmmlNoStats);
	// String evalFilePath = "src/test/resources/encog/nn/normalizedData";
	// EvalCSVUtil evalInput = new EvalCSVUtil(evalFilePath, pmmlNoStats);
	// List<Map<FieldName, String>> pmmlEvalResultList = evalInput
	// .getEvaluatorInput();
	// for (Map<FieldName, String> map : pmmlEvalResultList) {
	// @SuppressWarnings("unchecked")
	// Map<FieldName, Double> evalMap = (Map<FieldName, Double>) evaluator
	// .evaluate(map);
	// for (Map.Entry<FieldName, Double> entry : evalMap.entrySet()) {
	// System.out.println(entry.getValue());
	// }
	// }
	// }

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