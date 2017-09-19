/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.core.validator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * ModelInspectorTest class
 */
public class ModelInspectorTest {

    private File modelFile;
    private File columnFile;
    private ModelInspector instance;

    @BeforeClass
    public void setUp() throws IOException {
        File originalModel = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json");
        File originalColumn = new File(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");

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
        Assert.assertTrue(result.getStatus());

        config.getDataSet().setMetaColumnNameFile(" ");
        config.getDataSet().setCategoricalColumnNameFile("");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        config.getDataSet().setTargetColumnName(null);
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        config.getDataSet().setTargetColumnName("  ");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        config.getDataSet().setTargetColumnName("a");
        config.getDataSet().setMetaColumnNameFile("meta.names");
        FileUtils.write(new File("meta.names"), "a");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        FileUtils.write(new File("meta.names"), "");
        config.getDataSet().setTargetColumnName("a");
        config.getVarSelect().setForceRemoveColumnNameFile("force.remove.names");
        FileUtils.write(new File("force.remove.names"), "a");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        FileUtils.write(new File("force.remove.names"), "");
        config.getDataSet().setTargetColumnName("a");
        config.getVarSelect().setForceSelectColumnNameFile("force.select.names");
        FileUtils.write(new File("force.select.names"), "a");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        FileUtils.write(new File("force.select.names"), "");
        FileUtils.write(new File("force.remove.names"), "");
        FileUtils.write(new File("meta.names"), "b\nc");
        FileUtils.write(new File("force.select.names"), "c\nd");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        FileUtils.write(new File("force.select.names"), "");
        FileUtils.write(new File("force.remove.names"), "");
        FileUtils.write(new File("meta.names"), "b\nc");
        FileUtils.write(new File("force.remove.names"), "c\nd");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());

        FileUtils.write(new File("meta.names"), "");
        FileUtils.write(new File("force.remove.names"), "b\nc");
        FileUtils.write(new File("force.select.names"), "c\nd");
        result = instance.probe(config, ModelStep.INIT);
        Assert.assertTrue(result.getStatus());
        Assert.assertEquals("[true]", result.toString());

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

        config.getTrain().getParams().put(CommonConstants.NUM_HIDDEN_LAYERS, -1);
        result = instance.probe(config, ModelStep.TRAIN);
        Assert.assertFalse(result.getStatus());

        config.getTrain().getParams().put(CommonConstants.NUM_HIDDEN_LAYERS, 2);
        List<Integer> hiddenNodes = new ArrayList<Integer>();
        hiddenNodes.add(10);
        hiddenNodes.add(10);
        config.getTrain().getParams().put(CommonConstants.NUM_HIDDEN_NODES, hiddenNodes);
        List<String> activateFuncs = new ArrayList<String>();
        activateFuncs.add("tanh");
        activateFuncs.add("sigmoid");
        activateFuncs.add("tanh");
        config.getTrain().getParams().put(CommonConstants.ACTIVATION_FUNC, activateFuncs);
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
    public void tearDown() throws IOException {
        if(modelFile != null) {
            FileUtils.deleteQuietly(modelFile);
        }

        if(columnFile != null) {
            FileUtils.deleteQuietly(columnFile);
        }
    }
}
