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
package ml.shifu.shifu.pig;

import java.io.File;
import java.io.IOException;

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.io.FileUtils;
import org.powermock.modules.testng.PowerMockObjectFactory;
import org.testng.IObjectFactory;
import org.testng.annotations.ObjectFactory;


public class PigExecutorTest {

    @ObjectFactory
    public IObjectFactory setObjectFactor() {
        return new PowerMockObjectFactory();
    }

    
//    @Test
    public void test() throws IOException {
        PigExecutor exec = PigExecutor.getExecutor();
        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                SourceType.LOCAL);

        Environment.setProperty(Environment.SHIFU_HOME, ".");
        modelConfig.getDataSet().setSource(SourceType.LOCAL);
        exec.submitJob(modelConfig, "src/test/java/ml/shifu/shifu/pig/pigTest.pig");

        FileUtils.deleteQuietly(new File("ModelConfig.json"));
        FileUtils.deleteDirectory(new File("ModelSets"));
    }

}
