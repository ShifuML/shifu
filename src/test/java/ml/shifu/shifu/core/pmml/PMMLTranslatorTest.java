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
package ml.shifu.shifu.core.pmml;

import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.ShifuCLI;
import org.apache.commons.io.FileUtils;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ClassificationMap;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.NeuralNetworkEvaluator;
import org.powermock.modules.testng.PowerMockObjectFactory;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * ManagerTest class
 */
public class PMMLTranslatorTest {

    @ObjectFactory
    public IObjectFactory setObjectFactory() {
        return new PowerMockObjectFactory();
    }

    @BeforeClass
    public void setUp() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");
    }

    @Test
    public void testAllNumericVariablePmmlCase() throws Exception {

        // Step 1. Eval the scores using SHIFU
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        File modelsDir = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");
        File tmpModelsDir = new File("models");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        FileUtils.copyDirectory(modelsDir, tmpModelsDir);

        // run evaluation set
        ShifuCLI.runEvalSet(false);
        File evalScore = new File("evals/EvalA/EvalScore");

        exportPmml(originModel, originColumn, modelsDir);

        // Step 2. Eval the scores using PMML and compare it with SHIFU output

        String DataPath = "./src/test/resources/example/cancer-judgement/DataStore/Full_data/data.dat";
        String OutPath = "./pmml_out.dat";
        for (int index = 0; index < 5; index++) {
            String num = new Integer(index).toString();
            String pmmlPath = "pmmls/cancer-judgement" + num + ".pmml";
            evalPmml(pmmlPath, DataPath, OutPath, "\\|", "model" + num);
            compareScore(evalScore, new File(OutPath), "model" + num, "\\|", 1.0);
        }

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(tmpModelsDir);
        FileUtils.deleteQuietly(new File("./pmmls"));
        FileUtils.deleteQuietly(new File(OutPath));
    }


    @Test
    public void trainLaborNegModel() throws Exception {
        File originModel = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");
        File originColumn = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/ColumnConfig.json");
        FileUtils.copyFile(originModel, tmpModel);

        // shifu init
        ShifuCLI.initializeModel();
        File tmpColumn = new File("ColumnConfig.json");


        // shifu cal model stats
        long timestamp = tmpColumn.lastModified();
        ShifuCLI.calModelStats();
        Assert.assertTrue(tmpColumn.lastModified() > timestamp);

        // Shifu var selection
        timestamp = tmpColumn.lastModified();
        ShifuCLI.selectModelVar();
        Assert.assertTrue(tmpColumn.lastModified() > timestamp);

        // run normalization
        ShifuCLI.normalizeTrainData();
        File normalizedData = new File("tmp/NormalizedData");
        File selectedData = new File("tmp/SelectedRawData");
        Assert.assertTrue(normalizedData.exists());
        Assert.assertTrue(selectedData.exists());

        // run train
        ShifuCLI.trainModel(false, false);
        File tmpmodelFile = new File("models/model0.nn");
        Assert.assertTrue(tmpmodelFile.exists());
        FileUtils.copyFile(tmpColumn, originColumn);
        File modelFile = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/models/model0.nn");
        FileUtils.copyFile(tmpmodelFile, modelFile);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(new File("tmp"));
        FileUtils.deleteDirectory(new File("models"));
        FileUtils.deleteDirectory(new File("evals"));
    }


    @Test
    public void testMixTypeVariablePmmlCase() throws Exception {

        // Step 1. Eval the scores using SHIFU
        File originModel = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        File modelsDir = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/models");
        File tmpModelsDir = new File("models");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        FileUtils.copyDirectory(modelsDir, tmpModelsDir);

        // run evaluation set
        ShifuCLI.runEvalSet(false);
        File evalScore = new File("evals/EvalA/EvalScore");

        exportPmml(originModel, originColumn, modelsDir);

        // Step 2. Eval the scores using PMML
        String pmmlPath = "pmmls/ModelK0.pmml";
        String DataPath = "src/test/resources/example/labor-neg/DataStore/DataSet1/data.dat";
        String OutPath = "model_k_out.dat";
        evalPmml(pmmlPath, DataPath, OutPath, ",", "model0");

        // Step 3. Compare the SHIFU Eval score and PMML score
        compareScore(evalScore, new File(OutPath), "model0", "\\|", 1.0);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(tmpModelsDir);
        FileUtils.deleteQuietly(new File("./pmmls"));
        FileUtils.deleteQuietly(new File(OutPath));
        FileUtils.deleteQuietly(evalScore);
    }

    public void compareScore(File test, File control, String scoreName, String sep, Double error_range) throws Exception {
        List<String> testData = FileUtils.readLines(test);
        List<String> controlData = FileUtils.readLines(control);
        String[] testSchema = testData.get(0).trim().split(sep);
        String[] controlSchema = controlData.get(0).trim().split(sep);

        for (int row = 1; row < controlData.size(); row++) {
            Map<String, Object> ctx = new HashMap<String, Object>();
            Map<String, Object> controlCtx = new HashMap<String, Object>();

            String[] testRowValue = testData.get(row).split(sep, testSchema.length);
            for (int index = 0; index < testSchema.length; index++) {
                ctx.put(testSchema[index], testRowValue[index]);
            }
            String[] controlRowValue = controlData.get(row).split(sep, controlSchema.length);

            for (int index = 0; index < controlSchema.length; index++) {
                controlCtx.put(controlSchema[index], controlRowValue[index]);
            }
            Double controlScore = Double.valueOf((String) controlCtx.get(scoreName));
            Double testScore = Double.valueOf((String) ctx.get(scoreName));


            try {
                assert (controlScore - testScore < error_range && controlScore - testScore > -error_range);
            } catch (AssertionError e) {
                System.err.println(row + ": " + controlScore + "   " + testScore);
                e.printStackTrace();
                System.exit(-1);
            }
        }
    }


    public void exportPmml(File originModel, File originColumn, File modelsDir) throws Exception {

        File tmpModel = new File("ModelConfig.json");
        File tmpColumn = new File("ColumnConfig.json");


        File tmpModelsDir = new File("models");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        FileUtils.copyDirectory(modelsDir, tmpModelsDir);

        // run evaluation set
        ShifuCLI.exportModel(null);
    }

    public void evalPmml(String pmmlPath, String DataPath, String OutPath, String sep, String scoreName) throws Exception {

        PMML pmml = PMMLUtils.loadPMML(pmmlPath);
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(pmml);

        PrintWriter writer = new PrintWriter(OutPath, "UTF-8");
        writer.println(scoreName);
        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, DataPath, sep);

        for (Map<FieldName, FieldValue> maps : input) {
            switch (evaluator.getModel().getFunctionName()) {
                case REGRESSION:
                    Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                    for (Double value : regressionTerm.values()) {
                        System.out.println(value * 1000);
                        writer.println((int) Math.round(value * 1000));
                    }
                    break;
                case CLASSIFICATION:
                    Map<FieldName, ClassificationMap<String>> classificationTerm = (Map<FieldName, ClassificationMap<String>>) evaluator.evaluate(maps);
                    for (ClassificationMap<String> cMap : classificationTerm.values())
                        for (Map.Entry<String, Double> entry : cMap.entrySet())
                            System.out.println(entry.getValue() * 1000);
                default:
                    break;
            }
        }
        writer.close();
    }

    @AfterTest
    public void delete() throws IOException {
        FileUtils.deleteQuietly(new File("evals"));
    }
}
