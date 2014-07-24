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

import org.apache.mahout.classifier.mlp.MultilayerPerceptron;
import org.apache.mahout.classifier.mlp.NeuralNetwork;
import org.apache.mahout.math.DenseVector;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
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
public class PMMLMahoutNeuralNetworkTest {
	NeuralNetwork mlModel;
	PMML pmml;
	private static Logger log = LoggerFactory
			.getLogger(PMMLMahoutNeuralNetworkTest.class);
	NeuralNetworkEvaluator evaluator;
	protected final double DELTA = Math.pow(10, -5);
	private String inputData = "src/test/resources/data/wdbc/inputTrainData";
	private String initPmmlPath = "src/test/resources/data/wdbc/model.xml";
	private String outputPMMLPath = "src/test/resources/adapter/mahoutNN/mahoutNN.pmml";
	private String evalFilePath = "src/test/resources/data/wdbc/evalData";

	protected void initMLModel() {
		try {
			pmml = PMMLUtils.loadPMML(initPmmlPath);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		List<MahoutDataPair> inputDataSet = new MahoutTestDataGenerator(inputData, pmml)
				.getMahoutDataPair();
		// create Mahout NeuralNetwork model
		mlModel = new MultilayerPerceptron();
		mlModel.addLayer(30, false, "Identity");// inputFields,isFinalLayer,squashFunction
		mlModel.addLayer(45, false, "Sigmoid");
		mlModel.addLayer(45, false, "Sigmoid");
		mlModel.addLayer(1, true, "Sigmoid");

		for (MahoutDataPair pair : inputDataSet) {
			mlModel.trainOnline(pair.getVectorAsInputVector());
		}
	}

	protected void adaptToPMML() {
		Model pmmlNN = pmml.getModels().get(0);
		pmmlNN = new PMMLMahoutNeuralNetworkModel().adaptMLModelToPMML(mlModel,
				(org.dmg.pmml.NeuralNetwork) pmmlNN);
		pmml.getModels().set(0, pmmlNN);
	}

	protected void writeToPMML() {

		PMMLUtils.savePMML(pmml, outputPMMLPath);
		log.info(" - write PMML NeuralNetwork model to " + outputPMMLPath
				+ "\n - the number of nodes in each layer ...");
	}

	protected void evaluatePMML() {
		evaluator = new NeuralNetworkEvaluator(pmml);

		MahoutTestDataGenerator evalInput = new MahoutTestDataGenerator(evalFilePath, pmml);
		evaluateInputs(evalInput);

	}

	@Test
	public void testMahoutNN() {
		initMLModel();
		adaptToPMML();
		writeToPMML();
		evaluatePMML();
	}

	private void evaluateInputs(MahoutTestDataGenerator evalInput) {
		log.info(" evaluate  mahout neural network adapter with "
				+ evalInput.getEvaluatorInput().size() + " inputs");
		for (Map<FieldName, String> map : evalInput.getEvaluatorInput()) {
			ModelEvaluationContext context = new ModelEvaluationContext(null,
					evaluator);
			context.declareAll(map);
			double[] data = evalInput.normalizeData(context);
//			System.out.println("pmml " + getPMMLEvaluatorResult(map)
//					+ " mlModel "
//					+ mlModel.getOutput(new DenseVector(data)).get(0));
			Assert.assertEquals(getPMMLEvaluatorResult(map),
					mlModel.getOutput(new DenseVector(data)).get(0), DELTA);
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