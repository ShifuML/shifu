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
package ml.shifu.core.fs;

import ml.shifu.core.container.obj.ModelConfig;
import ml.shifu.core.container.obj.RawSourceData.SourceType;
import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.Environment;
import org.apache.commons.lang.NotImplementedException;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;


/**
 * PathFinderTest class
 */
public class PathFinderTest {

    private ModelConfig modelConfig;
    private PathFinder pathFinder;

    //@BeforeClass
    public void setUp() throws IOException {
        modelConfig = CommonUtils.loadModelConfig("src/test/resources/unittest/ModelSets/full/ModelConfig.json", SourceType.LOCAL);
        pathFinder = new PathFinder(modelConfig);
    }

    //@Test(expectedExceptions = IllegalArgumentException.class)
    public void testNullConstructor() {
        new PathFinder(null);
    }

    //@Test
    public void testGetModelConfigPath() {
        Assert.assertEquals(pathFinder.getModelConfigPath(SourceType.LOCAL), "ModelConfig.json");
        Assert.assertEquals(pathFinder.getModelConfigPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/ModelConfig.json");
    }

    //@Test(expectedExceptions = NotImplementedException.class)
    public void testGetModelConfigPathS3() {
        Assert.assertEquals(pathFinder.getModelConfigPath(SourceType.S3), "ModelConfig.json");
    }

    //@Test
    public void testGetColumnConfigPath() {
        Assert.assertEquals(pathFinder.getColumnConfigPath(SourceType.LOCAL), "ColumnConfig.json");
        Assert.assertEquals(pathFinder.getColumnConfigPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/ColumnConfig.json");
    }

   // @Test
    public void testGetAbsolutePath() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");

        Assert.assertEquals(pathFinder.getAbsolutePath("test"), "test");
        Assert.assertEquals(pathFinder.getAbsolutePath("/test"), "/test");
    }

    //@Test
    public void testGetJarPath() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");
        Assert.assertEquals(pathFinder.getJarPath(), "lib/*.jar");
    }

//	@Test
//	public void testGetReasonCodeMapPath() {
//		Environment.setProperty(Environment.SHIFU_HOME, ".");
//		Assert.assertEquals(pathFinder.getReasonCodeMapPath(SourceType.LOCAL), "common/ReasonCodeMapV3.json");
//		Assert.assertEquals(pathFinder.getReasonCodeMapPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/ReasonCodeMap.json");
//	}
//	
//	@Test(expectedExceptions = NotImplementedException.class)
//	public void testGetReasonCodeMapPathS3() {
//		Environment.setProperty(Environment.SHIFU_HOME, ".");
//		Assert.assertEquals(pathFinder.getReasonCodeMapPath(SourceType.S3), "common/ReasonCodeMapV3.json");
//	}
//	
//	@Test
//	public void testGetVariableStorePath() {
//		Environment.setProperty(Environment.SHIFU_HOME, ".");
//		Assert.assertEquals(pathFinder.getVariableStorePath(), "common/VariableStore.json");
//	}

    //@Test
    public void testGetNormalizedPath() {
        Assert.assertEquals(pathFinder.getEvalNormalizedPath(modelConfig.getEvalConfigByName("Eval1"), SourceType.LOCAL), "evals/Eval1/EvalNormalized");
        Assert.assertEquals(pathFinder.getEvalNormalizedPath(modelConfig.getEvalConfigByName("Eval1"), SourceType.HDFS), "ModelSets/UnitTestModelSet/evals/Eval1/EvalNormalized");
    }

    //@Test
    public void testGetEvalPath() {
        Assert.assertEquals(pathFinder.getEvalFilePath("Eval1", "EvalTester", SourceType.LOCAL), "evals/Eval1/EvalTester");
        Assert.assertEquals(pathFinder.getEvalFilePath("Eval1", "EvalTester", SourceType.HDFS), "ModelSets/UnitTestModelSet/evals/Eval1/EvalTester");
    }

    //@Test
    public void testGetEvalSetPath() {
        Assert.assertEquals(pathFinder.getEvalSetPath(modelConfig.getEvalConfigByName("Eval1"), SourceType.LOCAL), "evals/Eval1");
        Assert.assertEquals(pathFinder.getEvalSetPath(modelConfig.getEvalConfigByName("Eval1"), SourceType.HDFS), "ModelSets/UnitTestModelSet/evals/Eval1");

        Assert.assertEquals(pathFinder.getEvalSetPath("Eval1", SourceType.LOCAL), "evals/Eval1");
        Assert.assertEquals(pathFinder.getEvalSetPath("Eval1", SourceType.HDFS), "ModelSets/UnitTestModelSet/evals/Eval1");
    }

    //@Test
    public void testGetEvalScorePath() {
        Assert.assertEquals(pathFinder.getEvalScorePath(modelConfig.getEvalConfigByName("Eval1"), SourceType.LOCAL), "evals/Eval1/EvalScore");
        Assert.assertEquals(pathFinder.getEvalScorePath(modelConfig.getEvalConfigByName("Eval1"), SourceType.HDFS), "ModelSets/UnitTestModelSet/evals/Eval1/EvalScore");
    }

    //@Test
    public void testGetEvalPerformancePath() {
        Assert.assertEquals(pathFinder.getEvalPerformancePath(modelConfig.getEvalConfigByName("Eval1"), SourceType.LOCAL), "evals/Eval1/EvalPerformance.json");
        Assert.assertEquals(pathFinder.getEvalPerformancePath(modelConfig.getEvalConfigByName("Eval1"), SourceType.HDFS), "ModelSets/UnitTestModelSet/evals/Eval1/EvalPerformance.json");
    }
//	
//	@Test
//	public void testGetNetworksPath() {
//		Assert.assertEquals(pathFinder.getNetworksPath(SourceType.LOCAL), "models");
//		Assert.assertEquals(pathFinder.getNetworksPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/models");
//	}

    //@Test
    public void testGetNormalizedDataPath() {
        Assert.assertEquals(pathFinder.getNormalizedDataPath(SourceType.LOCAL), "tmp/NormalizedData");
        Assert.assertEquals(pathFinder.getNormalizedDataPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/tmp/NormalizedData");
    }

    //@Test(expectedExceptions = NotImplementedException.class)
    public void testGetNormalizedDataPathS3() {
        Assert.assertEquals(pathFinder.getNormalizedDataPath(SourceType.S3), "tmp/NormalizedData");
    }

    //@Test
    public void testGetPreTrainingStatsPath() {
        Assert.assertEquals(pathFinder.getPreTrainingStatsPath(SourceType.LOCAL), "tmp/PreTrainingStats");
        Assert.assertEquals(pathFinder.getPreTrainingStatsPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/tmp/PreTrainingStats");
    }

    //@Test
    public void testGetSelectedRawDataPath() {
        Assert.assertEquals(pathFinder.getSelectedRawDataPath(SourceType.LOCAL), "tmp/SelectedRawData");
        Assert.assertEquals(pathFinder.getSelectedRawDataPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/tmp/SelectedRawData");
    }

    //@Test
    public void testGetTrainScoresPath() {
        Assert.assertEquals(pathFinder.getTrainScoresPath(SourceType.LOCAL), "tmp/TrainScores");
        Assert.assertEquals(pathFinder.getTrainScoresPath(SourceType.HDFS), "ModelSets/UnitTestModelSet/tmp/TrainScores");
    }

    //@Test
    public void testGetBinAvgScorePath() {
        Assert.assertEquals(pathFinder.getBinAvgScorePath(SourceType.LOCAL), "tmp/BinAvgScore");
        Assert.assertEquals(pathFinder.getBinAvgScorePath(SourceType.HDFS), "ModelSets/UnitTestModelSet/tmp/BinAvgScore");
    }
}
