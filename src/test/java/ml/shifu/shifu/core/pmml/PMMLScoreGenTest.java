package ml.shifu.shifu.core.pmml;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Classification;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.TargetField;
import org.jpmml.evaluator.Value;
import org.jpmml.evaluator.mining.MiningModelEvaluator;
import org.testng.Assert;

import ml.shifu.shifu.ShifuCLI;
import ml.shifu.shifu.combo.CsvFile;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import ml.shifu.shifu.core.processor.EvalModelProcessor;
import ml.shifu.shifu.core.processor.ExportModelProcessor;

/**
 * Created by zhanhu on 2/8/17.
 */
public class PMMLScoreGenTest {

    public static final double EPS = 1e-6;

//    @Test

    public void testBaggingPmml() throws Exception {
        verifyPmml("TestNN", "src/test/resources/example/bagging-pmml/ModelConfig.json",
                "src/test/resources/example/bagging-pmml/ColumnConfig.json",
                "src/test/resources/example/bagging-pmml/columns", "src/test/resources/example/bagging-pmml/models",
                "src/test/resources/example/bagging-pmml/EvalSet1/eval.data.csv", "Eval1", "|");
    }

    private void verifyPmml(String modelName, String modelConfPath, String columnConfPath, String columnsPath,
            String modelsPath, String evalDataSet, String evalSetName, String delimiter) throws Exception {
        File tmpModel = new File("ModelConfig.json");
        FileUtils.copyFile(new File(modelConfPath), tmpModel);

        File tmpColumn = new File("ColumnConfig.json");
        FileUtils.copyFile(new File(columnConfPath), tmpColumn);

        File tmpNames = new File("columns");
        FileUtils.copyDirectory(new File(columnsPath), tmpNames);

        File tmpModels = new File("models");
        FileUtils.copyDirectory(new File(modelsPath), tmpModels);

        Map<String, Object> params = new HashMap<String, Object>();
        params.put(EvalModelProcessor.NOSORT, true);
        ShifuCLI.runEvalScore(evalSetName, params);

        genPMMLAndCompareScore(modelName, evalDataSet, evalSetName, delimiter);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(tmpNames);
        FileUtils.deleteDirectory(tmpModels);
        FileUtils.deleteQuietly(new File("evals"));
        FileUtils.deleteQuietly(new File("pmmls"));
    }

    private void genPMMLAndCompareScore(String modelName, String evalDataSet, String evalSetName, String delimiter)
            throws Exception {
        Map<String, Object> params = new HashMap<String, Object>();
        params.put(ExportModelProcessor.IS_CONCISE, true);
        ShifuCLI.exportModel(ExportModelProcessor.ONE_BAGGING_PMML_MODEL, params);

        int totalRecordCnt = 0;
        int matchRecordCnt = 0;

        CsvFile evalScoreFile = new CsvFile("evals" + File.separator + evalSetName + File.separator + "EvalScore", "|",
                true);
        Iterator<Map<String, String>> scoreIterator = evalScoreFile.iterator();
        scoreIterator.next(); // skip first line

        CsvFile evalData = new CsvFile(evalDataSet, delimiter, true);
        PMML pmml = PMMLUtils.loadPMML("pmmls" + File.separator + modelName + ".pmml");
        MiningModelEvaluator evaluator = new MiningModelEvaluator(pmml);
        Iterator<Map<String, String>> iterator = evalData.iterator();
        while(iterator.hasNext() && scoreIterator.hasNext()) {
            Map<String, String> rawInput = iterator.next();
            double pmmlScore = score(evaluator, rawInput, "FinalResult");

            Map<String, String> scoreInput = scoreIterator.next();
            double evalScore = Double.parseDouble(scoreInput.get("mean"));

            totalRecordCnt++;
            if(Math.abs(evalScore - pmmlScore) < EPS) {
                matchRecordCnt++;
            }
        }

        Assert.assertTrue(matchRecordCnt == totalRecordCnt);
    }

    @SuppressWarnings("unchecked")
    private double score(MiningModelEvaluator evaluator, Map<String, String> rawInput, String scoreName) {
        List<TargetField> targetFields = evaluator.getTargetFields();
        Map<FieldName, FieldValue> maps = convertRawIntoInput(evaluator, rawInput);
        List<Double> scores = new ArrayList<Double>();

        switch(evaluator.getModel().getMiningFunction()) {
            case REGRESSION:
                if(targetFields.size() == 1) {
                    Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                    scores.add(regressionTerm.get(new FieldName(AbstractSpecifCreator.FINAL_RESULT)));
                } else {
                    Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                    List<FieldName> outputFieldList = new ArrayList<FieldName>(regressionTerm.keySet());
                    Collections.sort(outputFieldList, new Comparator<FieldName>() {
                        @Override
                        public int compare(FieldName a, FieldName b) {
                            return a.getValue().compareTo(b.getValue());
                        }
                    });
                    for(int i = 0; i < outputFieldList.size(); i++) {
                        FieldName fieldName = outputFieldList.get(i);
                        if(fieldName.getValue().startsWith(AbstractSpecifCreator.FINAL_RESULT)) {
                            scores.add(regressionTerm.get(fieldName));
                        }
                    }
                }
                break;
            case CLASSIFICATION:
                Map<FieldName, Classification<Double>> classificationTerm = (Map<FieldName, Classification<Double>>) evaluator
                        .evaluate(maps);
                for(Classification<Double> cMap: classificationTerm.values())
                    for(Map.Entry<String, Value<Double>> entry: cMap.getValues().entrySet())
                        System.out.println(entry.getValue().getValue() * 1000);
                break;
            default:
                break;
        }

        return scores.get(0);
    }

    private Map<FieldName, FieldValue> convertRawIntoInput(MiningModelEvaluator evaluator,
            Map<String, String> rawInput) {
        Map<FieldName, FieldValue> arguments = new HashMap<FieldName, FieldValue>();
        for(InputField inputField: evaluator.getInputFields()) {
            FieldName name = inputField.getName();
            if(rawInput.containsKey(name.getValue())) {
                arguments.put(inputField.getName(), CsvUtil.prepare(inputField, rawInput.get(name.getValue())));
            } else {
                arguments.put(inputField.getName(), CsvUtil.prepare(inputField, null));
            }
        }

        return arguments;
    }
}
