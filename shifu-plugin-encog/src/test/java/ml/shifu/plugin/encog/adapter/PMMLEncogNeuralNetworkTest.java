package ml.shifu.plugin.encog.adapter;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Map;

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
        pmml = readPMMLFile(initPmmlPath);
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

        writePMML(outputPMMLPath, pmml);
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
        List<Map<FieldName, String>> pmmlEvalResultList = evalInput
                .getEvaluatorInput();
        log.info(" evaluate Encog NN adapter with " + pmmlEvalResultList.size()
                + " inputs");
        for (Map<FieldName, String> map : pmmlEvalResultList) {
            ModelEvaluationContext context = new ModelEvaluationContext(null,
                    evaluator);
            context.declareAll(map);
            MLData data = evalInput.normalizeData(context);
            Assert.assertEquals(getPMMLEvaluatorResult(map),
                    mlModel.compute(data).getData(0), DELTA);
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