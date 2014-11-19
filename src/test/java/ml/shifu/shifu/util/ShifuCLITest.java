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
package ml.shifu.shifu.util;

import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.core.pmml.PMMLUtils;

import org.apache.commons.io.FileUtils;
import org.dmg.pmml.PMML;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.ClassificationMap;
import org.jpmml.evaluator.FieldValue;
import org.easymock.EasyMock;
import org.jpmml.evaluator.NeuralNetworkEvaluator;
import org.powermock.api.easymock.PowerMock;
import org.powermock.modules.testng.PowerMockObjectFactory;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;


/**
 * ManagerTest class
 */
public class ShifuCLITest {

    @ObjectFactory
    public IObjectFactory setObjectFactory() {
        return new PowerMockObjectFactory();
    }

    @BeforeClass
    public void setUp() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");
    }

    // @Test
    public void testInitializeModelOld() throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(new File(
                "src/test/resources/data/ModelStore/ModelSet1/ModelConfig.json")));
        BufferedReader reader2 = new BufferedReader(new FileReader(new File(
                "src/test/resources/common/VariableStore.json")));
        String[] headers = "id|diagnosis|column_3|column_4|column_5|column_6|column_7|column_8|column_9|column_10|column_11|column_12|column_13|column_14|column_15|column_16|column_17|column_18|column_19|column_20|column_21|column_22|column_23|column_24|column_25|column_26|column_27|column_28|column_29|column_30|column_31|column_32|result"
                .split("\\|");

        PowerMock.mockStaticPartial(CommonUtils.class, "getReader", "getHeaders");

        EasyMock.expect(ShifuFileUtils.getReader("./ModelConfig.json", SourceType.LOCAL)).andReturn(reader).anyTimes();
        EasyMock.expect(ShifuFileUtils.getReader("common/VariableStore.json", SourceType.LOCAL)).andReturn(reader2)
                .anyTimes();
        EasyMock.expect(
                CommonUtils.getHeaders("./src/test/resources/data/DataStore/DataSet1/.pig_header", "|",
                        SourceType.LOCAL)
        ).andReturn(headers).anyTimes();

        PowerMock.replayAll(CommonUtils.class);

        ShifuCLI.initializeModel();

        File columnConfig = new File("./ColumnConfig.json");
        File tmp = new File("tmp");

        Assert.assertTrue(columnConfig.exists());
        Assert.assertTrue(tmp.exists());

        columnConfig.deleteOnExit();
        FileUtils.deleteDirectory(tmp);

        reader.close();
        reader2.close();
    }

    @Test
    public void testCreateModel() throws Exception {
        Environment.setProperty(Environment.SHIFU_HOME, "src/test/resources");
        ShifuCLI.createNewModel("TestModel", null, "It's a model for Unittest");

        File file = new File("TestModel");
        Assert.assertTrue(file.exists());
        FileUtils.deleteDirectory(file);
    }

    @Test
    public void testInitializeModel() throws Exception {
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        FileUtils.copyFile(originModel, tmpModel);

        ShifuCLI.initializeModel();

        File file = new File("ColumnConfig.json");
        Assert.assertTrue(file.exists());
        FileUtils.deleteQuietly(file);
        FileUtils.deleteQuietly(tmpModel);
    }

    @Test
    public void testCalculateModelStats() throws Exception {
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        long timestamp = tmpColumn.lastModified();

        ShifuCLI.initializeModel();
        ShifuCLI.calModelStats();

        Assert.assertTrue(tmpColumn.lastModified() > timestamp);
        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
    }

    @Test
    public void testSelectModelVar() throws Exception {
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);

        long timestamp = tmpColumn.lastModified();
        ShifuCLI.selectModelVar();
        Assert.assertTrue(tmpColumn.lastModified() > timestamp);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
    }

    @Test
    public void testNormalizeData() throws Exception {
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);

        ShifuCLI.initializeModel();
        ShifuCLI.calModelStats();
        ShifuCLI.normalizeTrainData();

        File normalizedData = new File("tmp/NormalizedData");
        File selectedData = new File("tmp/SelectedRawData");
        Assert.assertTrue(normalizedData.exists());
        Assert.assertTrue(selectedData.exists());

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(new File("tmp"));
    }

    @Test
    public void testTrainModel() throws Exception {
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");

        File originColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        // run normalization
        ShifuCLI.normalizeTrainData();

        // run train
        ShifuCLI.trainModel(false, false);

        File modelFile = new File("models/model0.nn");
        Assert.assertTrue(modelFile.exists());

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(new File("tmp"));
        FileUtils.deleteDirectory(new File("models"));

    }



    @Test
    public void testPostTrainModel() throws Exception {
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

        long timestamp = tmpColumn.lastModified();
        // run post-train
        ShifuCLI.initializeModel();
        ShifuCLI.calModelStats();
        ShifuCLI.selectModelVar();
        ShifuCLI.normalizeTrainData();
        ShifuCLI.postTrainModel();
        Assert.assertTrue(tmpColumn.lastModified() > timestamp);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(new File("tmp"));
        FileUtils.deleteDirectory(new File("models"));
    }

    @Test
    public void testRunEvalAll() throws Exception {
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
        Assert.assertTrue(evalScore.exists());

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteQuietly(new File("evals/EvalA/EvalConfusionMatrix"));
        FileUtils.deleteQuietly(new File("evals/EvalB/EvalConfusionMatrix"));
    }
    
    @Test
    public void test_ALL_numeric_variables_PMML_validation() throws Exception {
    	
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

        pmmlExport(originModel, originColumn, modelsDir);
        
        // Step 2. Eval the scores using PMML
        String PMMLFILEPATH = "cancer-judgement.pmml";
        String DataPath = "./src/test/resources/example/cancer-judgement/DataStore/Full_data/data.dat";
        String OutPath = "./pmml_out.dat";
        pmmlEvaluator(PMMLFILEPATH, DataPath, OutPath, "\\|");

        // Step 3. Compare the SHIFU Eval score and PMML score
        compare_score(evalScore, new File(OutPath), "model0", "\\|", 1.0);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(tmpModelsDir);
        FileUtils.deleteQuietly(new File(PMMLFILEPATH));
        FileUtils.deleteQuietly(new File(OutPath));
    }



    @Test
    public void testTrainModel_labor_neg() throws Exception {
        File originModel = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");
        File originColumn = new File("src/test/resources/example/labor-neg/DataStore/DataSet1/ColumnConfig.json");
        FileUtils.copyFile(originModel, tmpModel);

        // shifu init
        ShifuCLI.initializeModel();
        File file = new File("ColumnConfig.json");
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
    public void test_mix_type_variable_pmml_validation() throws Exception {

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

        pmmlExport(originModel, originColumn, modelsDir);

        // Step 2. Eval the scores using PMML
        String PMMLFILEPATH = "ModelK.pmml";
        String DataPath = "src/test/resources/example/labor-neg/DataStore/DataSet1/data.dat";
        String OutPath = "model_k_out.dat";
        pmmlEvaluator(PMMLFILEPATH, DataPath, OutPath, ",");

        // Step 3. Compare the SHIFU Eval score and PMML score
        compare_score(evalScore, new File(OutPath), "model0", "\\|", 1.0);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteDirectory(tmpModelsDir);
        FileUtils.deleteQuietly(new File(PMMLFILEPATH));
        FileUtils.deleteQuietly(new File(OutPath));
    }

    public void compare_score(File test, File control, String scoreName, String sep, Double error_range) throws Exception {
        List<String> testData = FileUtils.readLines(test);
        List<String> controlData = FileUtils.readLines(control);
        String[] testSchema = testData.get(0).trim().split(sep);
        String[] controlSchema = controlData.get(0).trim().split(sep);

        for(int row=1; row < controlData.size(); row++) {
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
                assert( controlScore - testScore < error_range && controlScore - testScore > -error_range);
            } catch (AssertionError e) {
                System.err.println(row + ": " + controlScore + "   " + testScore);
                e.printStackTrace();
                System.exit(-1);
            }
        }
    }


    public void pmmlExport(File originModel, File originColumn, File modelsDir) throws Exception {

        File tmpModel = new File("ModelConfig.json");
        File tmpColumn = new File("ColumnConfig.json");


        File tmpModelsDir = new File("models");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);
        FileUtils.copyDirectory(modelsDir, tmpModelsDir);

        // run evaluation set
        ShifuCLI.exportModel(null);
    }

    public void pmmlEvaluator(String PMMLFILEPATH, String DataPath, String OutPath, String sep) throws Exception {

        PMML pmml = PMMLUtils.loadPMML(PMMLFILEPATH);
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(pmml);

        PrintWriter writer = new PrintWriter(OutPath, "UTF-8");
        writer.println("model0");
        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, DataPath, sep);

        for (Map<FieldName, FieldValue> maps : input) {
            switch (evaluator.getModel().getFunctionName()) {
                case REGRESSION:
                    Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                    for (Double value : regressionTerm.values())
                    {
                        System.out.println(value * 1000);
                        writer.println((int)Math.round(value * 1000));
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

    @Test
    public void testCreateEvalSet() throws Exception {
        File originModel = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File tmpModel = new File("ModelConfig.json");
        File originColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        File tmpColumn = new File("ColumnConfig.json");

        FileUtils.copyFile(originModel, tmpModel);
        FileUtils.copyFile(originColumn, tmpColumn);

        long timestamp = tmpModel.lastModified();
        // run create evaluation set
        ShifuCLI.createNewEvalSet("EvalC");
        Assert.assertTrue(tmpModel.lastModified() > timestamp);

        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(new File("EvalC" + Constants.DEFAULT_EVALSCORE_META_COLUMN_FILE));
    }

    @Test
    public void testRunExport() throws Exception {
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
        ShifuCLI.exportModel(null);

        File pmml = new File("cancer-judgement.pmml");
        Assert.assertTrue(pmml.exists());
        
        FileUtils.deleteQuietly(tmpModel);
        FileUtils.deleteQuietly(tmpColumn);
        FileUtils.deleteQuietly(pmml);
        FileUtils.deleteDirectory(tmpModelsDir);
    }
    
    @AfterTest
    public void delete() throws IOException {
        FileUtils.deleteDirectory(new File("evals"));
    }

}
