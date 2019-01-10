package ml.shifu.shifu.util;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.FileStatus;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ModelSpecLoaderUtilsTest {

    @Test public void testFindModels() throws IOException {
        ModelConfig modelConfig = CommonUtils
                .loadModelConfig("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                        SourceType.LOCAL);

        File srcModels = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");
        File dstModels = new File("models");
        FileUtils.copyDirectory(srcModels, dstModels);

        List<FileStatus> modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, null, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        EvalConfig evalConfig = modelConfig.getEvalConfigByName("EvalA");
        evalConfig.setCustomPaths(new HashMap<String, String>());
        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, null);
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "  ");
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        FileUtils.deleteDirectory(dstModels);

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models/model0.nn");
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(1, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "not-exists");
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(0, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models/*.nn");
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet{0,1,9}/*/*.nn");
        modelFiles = ModelSpecLoaderUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());
    }

}
