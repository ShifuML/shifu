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
package ml.shifu.core.core.validator;

/**
 * ModelInspectorTest class
 */
public class ModelInspectorTest {
   /*
    private File modelFile;
    private File columnFile;
    private ModelInspector instance;

    @BeforeClass
    public void setUp() throws IOException {
        File originalModel = new File("src/test/resources/unittest/ModelSets/full/ModelConfig.json");
        File originalColumn = new File("src/test/resources/unittest/ModelSets/full/ColumnConfig.json");

        modelFile = new File("ModelConfig.json");
        columnFile = new File("ColumnConfig.json");

        FileUtils.copyFile(originalModel, modelFile);
        FileUtils.copyFile(originalColumn, columnFile);
        instance = ModelInspector.getInspector();
    }

    @Test
    public void testValidateMeta() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        instance.checkMeta(config);
    }

    @Test
    public void testValidateInit() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        config.getDataSet().setCategoricalColumnNameFile("not-exists");
        config.getDataSet().setMetaColumnNameFile("~/not-exists");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        config.getDataSet().setMetaColumnNameFile(" ");
        config.getDataSet().setCategoricalColumnNameFile("");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        config.getDataSet().setTargetColumnName(null);
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        config.getDataSet().setTargetColumnName("  ");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        config.getDataSet().setTargetColumnName("a");
        config.getDataSet().setMetaColumnNameFile("meta.names");
        FileUtils.write(new File("meta.names"), "a");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        FileUtils.write(new File("meta.names"), "");
        config.getDataSet().setTargetColumnName("a");
        config.getVarSelect().setForceRemoveColumnNameFile("force.remove.names");
        FileUtils.write(new File("force.remove.names"), "a");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        FileUtils.write(new File("force.remove.names"), "");
        config.getDataSet().setTargetColumnName("a");
        config.getVarSelect().setForceSelectColumnNameFile("force.select.names");
        FileUtils.write(new File("force.select.names"), "a");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        FileUtils.write(new File("force.select.names"), "");
        FileUtils.write(new File("force.remove.names"), "");
        FileUtils.write(new File("meta.names"), "b\nc");
        FileUtils.write(new File("force.select.names"), "c\nd");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        FileUtils.write(new File("force.select.names"), "");
        FileUtils.write(new File("force.remove.names"), "");
        FileUtils.write(new File("meta.names"), "b\nc");
        FileUtils.write(new File("force.remove.names"), "c\nd");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());

        FileUtils.write(new File("meta.names"), "");
        FileUtils.write(new File("force.remove.names"), "b\nc");
        FileUtils.write(new File("force.select.names"), "c\nd");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertFalse(result.getStatus());
        Assert.assertEquals("[false, Column - c exists both in force select conf and force remove conf.]", result.toString());

        FileUtils.deleteQuietly(new File("meta.names"));
        FileUtils.deleteQuietly(new File("force.remove.names"));
        FileUtils.deleteQuietly(new File("force.select.names"));
    }

    @Test
    public void testValidateStats() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.STATS);
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateVarSelect() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.VARSELECT);
        Assert.assertTrue(result.getStatus());

        config.getVarSelect().setForceRemoveColumnNameFile("not-exists");
        config.getVarSelect().setForceSelectColumnNameFile("~/not-exists");
        result = instance.probe(config, ModelStep.VARSELECT);
        Assert.assertFalse(result.getStatus());
    }

    @Test
    public void testValidateNormalize() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.NORMALIZE);
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateTrain() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.TRAIN);
        Assert.assertTrue(result.getStatus());

        config.getTrain().getGlobalParams().put(NNTrainer.NUM_HIDDEN_LAYERS, -1);
        result = instance.probe(config, ModelStep.TRAIN);
        Assert.assertFalse(result.getStatus());

        config.getTrain().getGlobalParams().put(NNTrainer.NUM_HIDDEN_LAYERS, 2);
        List<Integer> hiddenNodes = new ArrayList<Integer>();
        hiddenNodes.add(10);
        hiddenNodes.add(10);
        config.getTrain().getGlobalParams().put(NNTrainer.NUM_HIDDEN_NODES, hiddenNodes);
        List<String> activateFuncs = new ArrayList<String>();
        activateFuncs.add("tanh");
        activateFuncs.add("sigmod");
        activateFuncs.add("tanh");
        config.getTrain().getGlobalParams().put(NNTrainer.ACTIVATION_FUNC, activateFuncs);
        result = instance.probe(config, ModelStep.TRAIN);
        Assert.assertFalse(result.getStatus());
    }

    @Test
    public void testValidatePostTrain() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.POSTTRAIN);
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateEval() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, ModelStep.EVAL);
        Assert.assertTrue(result.getStatus());

        config.setEvals(null);
        result = instance.probe(config, ModelStep.EVAL);
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateNullStep() throws Exception {
        ModelConfig config = CommonUtils.loadModelConfig();
        ValidateResult result = instance.probe(config, null);
        Assert.assertTrue(result.getStatus());
    }

    @AfterClass
    public void tearDown() {
        if (modelFile != null) {
            modelFile.delete();
        }

        if (columnFile != null) {
            columnFile.delete();
        }
    }        */
}
