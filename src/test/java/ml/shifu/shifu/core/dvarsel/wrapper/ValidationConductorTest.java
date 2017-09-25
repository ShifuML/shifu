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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingRecord;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.testng.annotations.Test;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created on 11/27/2014.
 */
public class ValidationConductorTest {

    @Test
    public void testRunValidate() throws IOException {
        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                RawSourceData.SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                RawSourceData.SourceType.LOCAL);

        List<Integer> columnIdList = new ArrayList<Integer>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( columnConfig.isCandidate(hasCandidates) ) {
                columnIdList.add(columnConfig.getColumnNum());
            }
        }

        TrainingDataSet trainingDataSet = new TrainingDataSet(columnIdList);
        List<String> recordsList = IOUtils.readLines(
                new FileInputStream("src/test/resources/example/cancer-judgement/DataStore/DataSet1/part-00"));
        for( String record :  recordsList ) {
            addRecordIntoTrainDataSet(modelConfig, columnConfigList, trainingDataSet, record);
        }

        Set<Integer> workingList = new HashSet<Integer>();
        for ( Integer columnId : trainingDataSet.getDataColumnIdList() ) {
            workingList.clear();
            workingList.add(columnId);
            ValidationConductor conductor =
                    new ValidationConductor(modelConfig, columnConfigList, workingList, trainingDataSet);

            double error = conductor.runValidate();
            System.out.println("The error is - " + error + ", for columnId - " + columnId);
        }
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
            for ( Double normVal : normVals) {
                inputs[i++] = normVal;
            }
        }

        trainingDataSet.addTrainingRecord(new TrainingRecord(inputs, ideal, significance));
    }

    //@Test
    public void testPartershipModel() throws IOException {
        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "/Users/zhanhu/temp/partnership_varselect/ModelConfig.json",
                RawSourceData.SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "/Users/zhanhu/temp/partnership_varselect/ColumnConfig.json",
                RawSourceData.SourceType.LOCAL);

        List<Integer> columnIdList = new ArrayList<Integer>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( CommonUtils.isGoodCandidate(columnConfig, hasCandidates) ) {
                columnIdList.add(columnConfig.getColumnNum());
            }
        }

        TrainingDataSet trainingDataSet = new TrainingDataSet(columnIdList);
        List<String> recordsList = IOUtils.readLines(
                new FileInputStream("/Users/zhanhu/temp/partnership_varselect/part-m-00479"));
        for( String record :  recordsList ) {
            addNormalizedRecordIntoTrainDataSet(modelConfig, columnConfigList, trainingDataSet, record);
        }

        Set<Integer> workingList = new HashSet<Integer>();
        for ( Integer columnId : trainingDataSet.getDataColumnIdList() ) {
            workingList.clear();
            workingList.add(columnId);
            ValidationConductor conductor =
                    new ValidationConductor(modelConfig, columnConfigList, workingList, trainingDataSet);

            double error = conductor.runValidate();
            System.out.println("The error is - " + error + ", for columnId - " + columnId);
        }
    }

    public void addNormalizedRecordIntoTrainDataSet(ModelConfig modelConfig,
                                          List<ColumnConfig> columnConfigList,
                                          TrainingDataSet trainingDataSet,
                                          String record) {
        String[] fields = CommonUtils.split(record, "|");

        double[] inputs = new double[trainingDataSet.getDataColumnIdList().size()];
        double[] ideal = new double[1];

        double significance = NNConstants.DEFAULT_SIGNIFICANCE_VALUE;

        int targetColumnId = CommonUtils.getTargetColumnNum(columnConfigList);
        ideal[0] = Double.parseDouble(fields[targetColumnId]);

        int i = 0;
        for ( Integer columnId : trainingDataSet.getDataColumnIdList() ) {
            if ( StringUtils.isBlank(fields[columnId]) ) {
                System.out.println(columnId + "|" + fields[columnId]);
            }

            try {
                inputs[i++] = Double.parseDouble(fields[columnId]);
            } catch ( Exception e ) {
                System.out.println(columnId + "|" + fields[columnId]);
            }
        }

        trainingDataSet.addTrainingRecord(new TrainingRecord(inputs, ideal, significance));
    }

}
