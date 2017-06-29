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
package ml.shifu.shifu.core.module;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.PerformanceEvaluator;
import org.apache.commons.io.FileUtils;
import org.testng.annotations.AfterTest;
import org.testng.annotations.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class PerformanceEvaluatorTest {

    @Test(expectedExceptions = FileNotFoundException.class)
    public void reviewTest() throws IOException {
        ModelConfig model = ModelConfig.createInitModelConfig("test", ALGORITHM.NN, ".", false);
        EvalConfig eval = new EvalConfig();
        eval.setName("test");
        eval.setDataSet(new RawSourceData());

        model.getBasic().setRunMode(RunMode.LOCAL);
        PerformanceEvaluator actor = new PerformanceEvaluator(model, eval);

        actor.review();
    }

    @AfterTest
    public void tearDown() throws IOException {
        FileUtils.deleteDirectory(new File("test"));
    }
}
