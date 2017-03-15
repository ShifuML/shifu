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
package ml.shifu.shifu.fs;

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;
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

    @BeforeClass
    public void setUp() throws IOException {
        modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json", SourceType.LOCAL);
        pathFinder = new PathFinder(modelConfig);
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testNullConstructor() {
        new PathFinder(null);
    }

    @Test
    public void testGetModelConfigPath() {
        Assert.assertEquals(pathFinder.getModelConfigPath(SourceType.LOCAL), "ModelConfig.json");
        Assert.assertTrue(pathFinder.getModelConfigPath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/ModelConfig.json"));
    }

    @Test(expectedExceptions = NotImplementedException.class)
    public void testGetModelConfigPathS3() {
        Assert.assertEquals(pathFinder.getModelConfigPath(SourceType.S3), "ModelConfig.json");
    }

    @Test
    public void testGetColumnConfigPath() {
        Assert.assertEquals(pathFinder.getColumnConfigPath(SourceType.LOCAL), "ColumnConfig.json");
        Assert.assertTrue(pathFinder.getColumnConfigPath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/ColumnConfig.json"));
    }

    @Test
    public void testGetAbsolutePath() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");

        Assert.assertEquals(pathFinder.getScriptPath("test"), "test");
        Assert.assertEquals(pathFinder.getScriptPath("/test"), "/test");
    }

    @Test
    public void testGetJarPath() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");
        Assert.assertEquals(pathFinder.getJarPath(), "lib/*.jar");
    }

    // @Test
    // public void testGetReasonCodeMapPath() {
    // Environment.setProperty(Environment.SHIFU_HOME, ".");
    // Assert.assertEquals(pathFinder.getReasonCodeMapPath(SourceType.LOCAL), "common/ReasonCodeMapV3.json");
    // Assert.assertEquals(pathFinder.getReasonCodeMapPath(SourceType.HDFS),
    // "ModelSets/cancer-judgement/ReasonCodeMap.json");
    // }
    //
    // @Test(expectedExceptions = NotImplementedException.class)
    // public void testGetReasonCodeMapPathS3() {
    // Environment.setProperty(Environment.SHIFU_HOME, ".");
    // Assert.assertEquals(pathFinder.getReasonCodeMapPath(SourceType.S3), "common/ReasonCodeMapV3.json");
    // }
    //
    // @Test
    // public void testGetVariableStorePath() {
    // Environment.setProperty(Environment.SHIFU_HOME, ".");
    // Assert.assertEquals(pathFinder.getVariableStorePath(), "common/VariableStore.json");
    // }

    @Test
    public void testGetNormalizedPath() {
        Assert.assertEquals(
                pathFinder.getEvalNormalizedPath(modelConfig.getEvalConfigByName("EvalA"), SourceType.LOCAL),
                "evals/EvalA/EvalNormalized");
        Assert.assertTrue(pathFinder.getEvalNormalizedPath(modelConfig.getEvalConfigByName("EvalA"), SourceType.HDFS)
                .contains("ModelSets/cancer-judgement/evals/EvalA/EvalNormalized"));
    }

    @Test
    public void testGetEvalPath() {
        Assert.assertEquals(pathFinder.getEvalFilePath("EvalA", "EvalTester", SourceType.LOCAL),
                "evals/EvalA/EvalTester");
        Assert.assertTrue(pathFinder.getEvalFilePath("EvalA", "EvalTester", SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/evals/EvalA/EvalTester"));
    }

    @Test
    public void testGetEvalSetPath() {
        Assert.assertEquals(pathFinder.getEvalSetPath(modelConfig.getEvalConfigByName("EvalA"), SourceType.LOCAL),
                "evals/EvalA");
        Assert.assertTrue(pathFinder.getEvalSetPath(modelConfig.getEvalConfigByName("EvalA"), SourceType.HDFS)
                .contains("ModelSets/cancer-judgement/evals/EvalA"));

        Assert.assertEquals(pathFinder.getEvalSetPath("EvalA", SourceType.LOCAL), "evals/EvalA");
        Assert.assertTrue(pathFinder.getEvalSetPath("EvalA", SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/evals/EvalA"));
    }

    @Test
    public void testGetEvalScorePath() {
        Assert.assertEquals(pathFinder.getEvalScorePath(modelConfig.getEvalConfigByName("EvalA"), SourceType.LOCAL),
                "evals/EvalA/EvalScore");
        Assert.assertTrue(pathFinder.getEvalScorePath(modelConfig.getEvalConfigByName("EvalA"), SourceType.HDFS)
                .contains("ModelSets/cancer-judgement/evals/EvalA/EvalScore"));
    }

    @Test
    public void testGetEvalPerformancePath() {
        Assert.assertEquals(
                pathFinder.getEvalPerformancePath(modelConfig.getEvalConfigByName("EvalA"), SourceType.LOCAL),
                "evals/EvalA/EvalPerformance.json");
        Assert.assertTrue(pathFinder.getEvalPerformancePath(modelConfig.getEvalConfigByName("EvalA"), SourceType.HDFS)
                .contains("ModelSets/cancer-judgement/evals/EvalA/EvalPerformance.json"));
    }

    //
    // @Test
    // public void testGetNetworksPath() {
    // Assert.assertEquals(pathFinder.getNetworksPath(SourceType.LOCAL), "models");
    // Assert.assertEquals(pathFinder.getNetworksPath(SourceType.HDFS), "ModelSets/cancer-judgement/models");
    // }

    @Test
    public void testGetNormalizedDataPath() {
        Assert.assertEquals(pathFinder.getNormalizedDataPath(SourceType.LOCAL), "tmp/NormalizedData");
        Assert.assertTrue(pathFinder.getNormalizedDataPath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/tmp/NormalizedData"));
    }

    @Test(expectedExceptions = NotImplementedException.class)
    public void testGetNormalizedDataPathS3() {
        Assert.assertEquals(pathFinder.getNormalizedDataPath(SourceType.S3), "tmp/NormalizedData");
    }

    @Test
    public void testGetPreTrainingStatsPath() {
        Assert.assertEquals(pathFinder.getPreTrainingStatsPath(SourceType.LOCAL), "tmp/PreTrainingStats");
        Assert.assertTrue(pathFinder.getPreTrainingStatsPath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/tmp/PreTrainingStats"));
    }

    @Test
    public void testGetSelectedRawDataPath() {
        Assert.assertEquals(pathFinder.getSelectedRawDataPath(SourceType.LOCAL), "tmp/SelectedRawData");
        Assert.assertTrue(pathFinder.getSelectedRawDataPath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/tmp/SelectedRawData"));
    }

    @Test
    public void testGetTrainScoresPath() {
        Assert.assertEquals(pathFinder.getTrainScoresPath(SourceType.LOCAL), "tmp/TrainScores");
        Assert.assertTrue(pathFinder.getTrainScoresPath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/tmp/TrainScores"));
    }

    @Test
    public void testGetBinAvgScorePath() {
        Assert.assertEquals(pathFinder.getBinAvgScorePath(SourceType.LOCAL), "tmp/BinAvgScore");
        Assert.assertTrue(pathFinder.getBinAvgScorePath(SourceType.HDFS).contains(
                "ModelSets/cancer-judgement/tmp/BinAvgScore"));
    }
}
