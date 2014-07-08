package ml.shifu.plugin.encog.adapter;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Map;

import ml.shifu.core.plugin.pmml.AdapterConstants;
import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelTest;
import ml.shifu.core.util.PMMLUtils;

import org.dmg.pmml.Constant;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
import org.jpmml.evaluator.NeuralNetworkEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * Test PMMLEncogNeuralNetworkModel that converts an Encog NeuralNetwork model
 * to a PMML NeuralNetwork Model.
 */
public class PMMLEncogNeuralNetworkTest extends PMMLModelTest<BasicNetwork> {
	BasicNetwork mlModel;
	PMML pmml;
	private static Logger log = LoggerFactory
			.getLogger(PMMLEncogNeuralNetworkTest.class);
	NeuralNetworkEvaluator evaluator;
	protected final double DELTA = Math.pow(10, -5);
	private String mlModelPath = "src/test/resources/encog/nn/EncogNN.nn";
	private String initPmmlPath = "src/test/resources/encog/nn/model.xml";
	private String outputPMMLPath = "src/test/resources/encog/nn/EncogNN_output.pmml";

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
		String evalFilePath = "src/test/resources/encog/nn/wdbc.train";
		EvalCSVUtil evalInput = new EvalCSVUtil(evalFilePath, pmml);
		evaluateInputs(evalInput);

		// evalFilePath =
		// "src/test/resources/data/wdbc/inputField20/evalData560.csv";
		// evalInput = new EvalCSVUtil(evalFilePath, headers);
		// evaluateInputs(evalInput);
	}

	@Test
	public void testEncogNN_2layer() {
		testSetUp();
		writeToPMML();
		evaluatePMML();
	}

	private void evaluateInputs(EvalCSVUtil evalInput) {
		// List<Map<FieldName, String>> pmmlEvalResultList = evalInput
		// .getEvaluatorInput();
		// evaluateNormalizedData();
		// log.info(" evaluate Encog NN adapter with " +
		// pmmlEvalResultList.size()
		// + " inputs");
		// for (Map<FieldName, String> map : pmmlEvalResultList) {
		// ModelEvaluationContext context = new ModelEvaluationContext(null,
		// evaluator);
		// context.declareAll(map);
		// MLData data = evalInput.normalizeData(context);
		// log.info("," + mlModel.compute(data).getData(0));
		// System.out.println( getPMMLEvaluatorResult(map));
		// }

	}

	@SuppressWarnings("unused")
	private void evaluateNormalizedData() throws Exception {

		PMML pmmlNoStats = PMMLUtils.loadPMML(outputPMMLPath);
		NeuralNetwork pmmlNN = (NeuralNetwork) pmmlNoStats.getModels().get(0);
		List<String> activeFields = PMMLAdapterCommonUtil
				.getSchemaActiveFields(pmmlNN.getMiningSchema());
		DataDictionary dictionary = new DataDictionary();
		for (String field : activeFields) {
			DataField targetField = new DataField(new FieldName(field),
					OpType.CONTINUOUS, DataType.DOUBLE);
			// targetField.withValues(new Value("1")).withValues(new
			// Value("0"));
			dictionary.withDataFields(targetField);
		}
		pmmlNoStats.setDataDictionary(dictionary);
		pmmlNN.setModelStats(null);

		DerivedField field = new DerivedField(OpType.CONTINUOUS,
				DataType.DOUBLE).withName(new FieldName(
				AdapterConstants.biasValue));
		// field.withName(new FieldName(s));
		field.withExpression(new Constant(String.valueOf(AdapterConstants.bias)));
		pmmlNN.setLocalTransformations(new LocalTransformations()
				.withDerivedFields(field));
		pmmlNoStats.withModels(pmmlNN);
		// rebuild data dictionary
		// copy mining schema
		PMMLUtils.savePMML(pmmlNoStats,
				"src/test/resources/encog/nn/EncogNN_noStats.pmml");
		NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(
				pmmlNoStats);
		String evalFilePath = "src/test/resources/encog/nn/normalizedData";
		EvalCSVUtil evalInput = new EvalCSVUtil(evalFilePath, pmmlNoStats);
		List<Map<FieldName, String>> pmmlEvalResultList = evalInput
				.getEvaluatorInput();
		for (Map<FieldName, String> map : pmmlEvalResultList) {
			@SuppressWarnings("unchecked")
			Map<FieldName, Double> evalMap = (Map<FieldName, Double>) evaluator
					.evaluate(map);
			for (Map.Entry<FieldName, Double> entry : evalMap.entrySet()) {
				System.out.println(entry.getValue());
			}
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