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
package ml.shifu.shifu.core;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.Constants;

/**
 * ConfusionMatrixTest class
 */
public class ConfusionMatrixTest {

    private ModelConfig modelConfig;
    private EvalConfig evalConfig;

    @BeforeClass
    public void setUp() throws IOException {
        modelConfig = ModelConfig.createInitModelConfig("test", ALGORITHM.NN, null, false);
        evalConfig = modelConfig.getEvalConfigByName("Eval1");
        new File("./models").mkdir();
    }

    @Test(expectedExceptions = ShifuException.class)
    public void testEvalScoreColumnNull() throws IOException {
        Map<String, String> customPaths = new HashMap<String, String>();
        customPaths.put(Constants.KEY_SCORE_PATH, "src/test/resources/example/cancer-judgement/DataStore/EvalSet1");
        evalConfig.setCustomPaths(customPaths);

        evalConfig.setPerformanceScoreSelector(null);
        new ConfusionMatrix(modelConfig, evalConfig, this);
    }

    @Test(expectedExceptions = ShifuException.class)
    public void testEvalScoreColumnNotFound() throws IOException {
        Map<String, String> customPaths = new HashMap<String, String>();
        customPaths.put(Constants.KEY_SCORE_PATH, "src/test/resources/example/cancer-judgement/DataStore/EvalSet1");
        evalConfig.setCustomPaths(customPaths);

        evalConfig.setPerformanceScoreSelector("mean");
        new ConfusionMatrix(modelConfig, evalConfig, this);
    }

    @Test
    public void testEvalScoreColumnFound() throws IOException {
        Map<String, String> customPaths = new HashMap<String, String>();
        customPaths.put(Constants.KEY_SCORE_PATH, "src/test/resources/example/cancer-judgement/DataStore/EvalSet1");
        evalConfig.setCustomPaths(customPaths);

        evalConfig.setPerformanceScoreSelector("diagnosis");
        new ConfusionMatrix(modelConfig, evalConfig, this);
    }

    @Test(expectedExceptions = { ShifuException.class, FileNotFoundException.class })
    public void testEvalScoreHeaderNotExists() throws IOException {
        Map<String, String> customPaths = new HashMap<String, String>();
        customPaths.put(Constants.KEY_SCORE_PATH, "src/test/resources/data/dt/models");
        evalConfig.setCustomPaths(customPaths);

        evalConfig.setPerformanceScoreSelector(null);
        new ConfusionMatrix(modelConfig, evalConfig, this);
    }

    @Test(expectedExceptions = ShifuException.class)
    public void testEvalTargetColumnNotFound() throws IOException {
        Map<String, String> customPaths = new HashMap<String, String>();
        customPaths.put(Constants.KEY_SCORE_PATH, "src/test/resources/example/cancer-judgement/DataStore/EvalSet1");
        evalConfig.setCustomPaths(customPaths);

        evalConfig.setPerformanceScoreSelector("diagnosis");
        evalConfig.getDataSet().setTargetColumnName("xxxx");
        new ConfusionMatrix(modelConfig, evalConfig, this);
    }

    @Test
    public void testEvalTargetColumnFound() throws IOException {
        Map<String, String> customPaths = new HashMap<String, String>();
        customPaths.put(Constants.KEY_SCORE_PATH, "src/test/resources/example/cancer-judgement/DataStore/EvalSet1");
        evalConfig.setCustomPaths(customPaths);

        evalConfig.setPerformanceScoreSelector("diagnosis");
        evalConfig.getDataSet().setTargetColumnName("diagnosis");
        new ConfusionMatrix(modelConfig, evalConfig);
    }

    @AfterClass
    public void tearDown() throws IOException {
        File dir = new File("test");
        FileUtils.deleteDirectory(dir);
        FileUtils.deleteDirectory(new File("./models"));
    }
}
