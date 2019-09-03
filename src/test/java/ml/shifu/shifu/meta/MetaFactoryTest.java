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
package ml.shifu.shifu.meta;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.meta.MetaFactory;
import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.dtrain.CommonConstants;

import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * MetaFactoryTest class
 */
public class MetaFactoryTest {

    private ModelConfig modelConfig;

    @BeforeClass
    public void setUp() throws IOException {
        modelConfig = ModelConfig.createInitModelConfig("unittest", ALGORITHM.NN, "a model config for unit-test", false);
        modelConfig.getBasic().setName("TestMode");
        modelConfig.getBasic().setAuthor("Author");
    }

    @Test
    public void testGetModelConfigMeta() {
        Assert.assertNotNull(MetaFactory.getModelConfigMeta().size());
    }

    @Test
    public void testValidateModelConfig() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig);
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelBasicConf() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getBasic());
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelSouceDataConf() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getDataSet());
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelStatsConf() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getStats());
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelVarSelectConf() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getVarSelect());
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelNormalizeConf() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getNormalize());
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelTrainAConf() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getTrain());
        Assert.assertTrue(result.getStatus());

        Map<String, Object> originalParams = modelConfig.getTrain().getParams();

        Map<String, Object> params = new HashMap<String, Object>();
        List<Object> nodesList = new ArrayList<Object>();
        nodesList.add("a");
        nodesList.add(45);

        params.put(CommonConstants.NUM_HIDDEN_NODES, nodesList);
        modelConfig.getTrain().setParams(params);
        result = MetaFactory.validate(modelConfig.getTrain());
        Assert.assertFalse(result.getStatus());
        modelConfig.getTrain().setParams(originalParams);
    }

    @Test
    public void testValidateModelEvalList() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getEvals());
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelEval() throws Exception {
        ValidateResult result = MetaFactory.validate(modelConfig.getEvals().get(0));
        Assert.assertTrue(result.getStatus());
    }

    @Test
    public void testValidateModelTrainBConf() throws Exception {
        ModelTrainConf trainConf = new ModelTrainConf();
        trainConf.setAlgorithm("test");
        trainConf.setBaggingNum(10);
        trainConf.setTrainOnDisk(null);

        ValidateResult result = MetaFactory.validate(trainConf);
        Assert.assertFalse(result.getStatus());
    }

    @AfterClass
    public void tearDown() throws IOException {
        FileUtils.deleteDirectory(new File("unittest"));
    }

}
