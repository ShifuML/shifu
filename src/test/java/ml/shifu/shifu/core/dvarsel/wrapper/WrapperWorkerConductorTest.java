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
package ml.shifu.shifu.core.dvarsel.wrapper;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import junit.framework.Assert;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dvarsel.CandidateSeed;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingRecord;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.testng.annotations.Test;

/**
 * Created on 11/27/2014.
 */
public class WrapperWorkerConductorTest {

    @Test
    public void testWrapperConductor() throws IOException {
        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                RawSourceData.SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                RawSourceData.SourceType.LOCAL);

        WrapperWorkerConductor wrapper = new WrapperWorkerConductor(modelConfig, columnConfigList);
        TrainingDataSet trainingDataSet = genTrainingDataSet(modelConfig, columnConfigList);
        wrapper.retainData(trainingDataSet);

        List<Integer> columnIdList = new ArrayList<Integer>();
        for ( int i = 2; i < 30; i ++ ) {
            columnIdList.add(i);
        }

        List<CandidateSeed> seedList = new ArrayList<CandidateSeed>();
        for ( int i = 0; i < 10; i ++ ) {
            seedList.add(new CandidateSeed(0, columnIdList.subList(i + 1, i + 7)));
        }
        wrapper.consumeMasterResult(new VarSelMasterResult(seedList));
        VarSelWorkerResult workerResult = wrapper.generateVarSelResult();

        Assert.assertNotNull(workerResult);
        Assert.assertTrue(workerResult.getSeedPerfList().size() > 0 );
    }

    public TrainingDataSet genTrainingDataSet(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) throws IOException {
        List<Integer> columnIdList = new ArrayList<Integer>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for (ColumnConfig columnConfig : columnConfigList) {
            if (columnConfig.isCandidate(hasCandidates)) {
                columnIdList.add(columnConfig.getColumnNum());
            }
        }

        TrainingDataSet trainingDataSet = new TrainingDataSet(columnIdList);
        List<String> recordsList = IOUtils.readLines(
                new FileInputStream("src/test/resources/example/cancer-judgement/DataStore/DataSet1/part-00"));
        for (String record : recordsList) {
            addRecordIntoTrainDataSet(modelConfig, columnConfigList, trainingDataSet, record);
        }

        return trainingDataSet;
    }

    public void addRecordIntoTrainDataSet(ModelConfig modelConfig,
                                          List<ColumnConfig> columnConfigList,
                                          TrainingDataSet trainingDataSet,
                                          String record) {
        String[] fields = CommonUtils.split(record, modelConfig.getDataSetDelimiter());

        int targetColumnId = CommonUtils.getTargetColumnNum(columnConfigList);
        String tag = StringUtils.trim(fields[targetColumnId]);

        double[] inputs = new double[trainingDataSet.getDataColumnIdList().size()];
        double[] ideal = new double[1];

        double significance = CommonConstants.DEFAULT_SIGNIFICANCE_VALUE;

        ideal[0] = (modelConfig.getPosTags().contains(tag) ? 1.0d : 0.0d);

        int i = 0;
        for ( Integer columnId : trainingDataSet.getDataColumnIdList() ) {
            List<Double> normVals = Normalizer.normalize(columnConfigList.get(columnId), fields[columnId]);
            for (Double normVal : normVals) {
                inputs[i++] = normVal;
            }
        }

        trainingDataSet.addTrainingRecord(new TrainingRecord(inputs, ideal, significance));
    }
}
