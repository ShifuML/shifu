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
package ml.shifu.shifu.core.dvarsel;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingRecord;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created on 11/24/2014.
 */
public class VarSelWorker
        extends
        AbstractWorkerComputable<VarSelMasterResult, VarSelWorkerResult, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {
    private static final Logger LOG = LoggerFactory.getLogger(VarSelWorker.class);

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;

    private AbstractWorkerConductor workerConductor;

    private long count = 0;
    private int inputNodeCount;
    private int outputNodeCount;

    private DataPurifier dataPurifier;
    private int targetColumnId = -1;
    private int weightColumnId = -1;

    private TrainingDataSet trainingDataSet;
    private long posRecordCount = 0;
    private long totalRecordCount = 0;

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader());
        this.getRecordReader().initialize(fileSplit);
    }

    @Override
    public void init(WorkerContext<VarSelMasterResult, VarSelWorkerResult> workerContext) {
        Properties props = workerContext.getProps();

        try {
            RawSourceData.SourceType sourceType = RawSourceData.SourceType.valueOf(props.getProperty(
                    CommonConstants.MODELSET_SOURCE_TYPE, RawSourceData.SourceType.HDFS.toString()));

            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);

            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);

            String conductorClsName = props.getProperty(Constants.VAR_SEL_WORKER_CONDUCTOR);
            this.workerConductor = (AbstractWorkerConductor) Class.forName(conductorClsName)
                    .getDeclaredConstructor(ModelConfig.class, List.class)
                    .newInstance(this.modelConfig, this.columnConfigList);
        } catch (IOException e) {
            throw new RuntimeException("Fail to load ModelConfig or List<ColumnConfig>", e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Invalid Master Conductor class", e);
        } catch (InstantiationException e) {
            throw new RuntimeException("Fail to create instance", e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException("Illegal access when creating instance", e);
        } catch (NoSuchMethodException e) {
            throw new RuntimeException("Fail to call method when creating instance", e);
        } catch (InvocationTargetException e) {
            throw new RuntimeException("Fail to invoke when creating instance", e);
        }

        List<Integer> normalizedColumnIdList = this.getNormalizedColumnIdList();
        this.inputNodeCount = normalizedColumnIdList.size();
        this.outputNodeCount = this.getTargetColumnCount();

        trainingDataSet = new TrainingDataSet(normalizedColumnIdList);
        try {
            dataPurifier = new DataPurifier(modelConfig);
        } catch (IOException e) {
            throw new RuntimeException("Fail to create DataPurifier", e);
        }

        this.targetColumnId = CommonUtils.getTargetColumnNum(this.columnConfigList);
        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())) {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.getColumnName().equalsIgnoreCase(modelConfig.getWeightColumnName().trim())) {
                    this.weightColumnId = columnConfig.getColumnNum();
                    break;
                }
            }
        }
    }

    @Override
    public VarSelWorkerResult doCompute(WorkerContext<VarSelMasterResult, VarSelWorkerResult> workerContext) {
        if(!workerConductor.isInitialized()) {
            LOG.info("There are {} records in current worker, with {} positive records.", totalRecordCount,
                    posRecordCount);
            workerConductor.retainData(trainingDataSet);
        }

        VarSelMasterResult masterResult = workerContext.getLastMasterResult();
        if(masterResult == null) {
            // no working set, wait master to send the working set
            return workerConductor.getDefaultWorkerResult();
        }

        if(masterResult.isHalt()) {
            // finish variable selection, stop working
            return null;
        }

        LOG.info("Get result from master, the base seed count is - {}", masterResult.getSeedList().size());

        workerConductor.consumeMasterResult(masterResult);
        return workerConductor.generateVarSelResult();
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<VarSelMasterResult, VarSelWorkerResult> workerContext) {
        if((++this.count) % 100000 == 0) {
            LOG.info("Read {} records.", this.count);
        }
        String record = currentValue.getWritable().toString();
        String[] fields = CommonUtils.split(record, this.modelConfig.getDataSetDelimiter());
        String tag = CommonUtils.trimTag(fields[this.targetColumnId]);

        if(this.dataPurifier.isFilter(record) && isPosOrNegTag(this.modelConfig, tag)) {
            this.totalRecordCount++;
            if(this.modelConfig.getPosTags().contains(tag)) {
                this.posRecordCount++;
            }

            double[] inputs = new double[this.inputNodeCount];
            double[] ideal = new double[this.outputNodeCount];

            double significance = CommonConstants.DEFAULT_SIGNIFICANCE_VALUE;
            if(this.weightColumnId >= 0) {
                try {
                    significance = Double.parseDouble(fields[this.weightColumnId]);
                } catch (Exception e) {
                    // user may set wrong field, just used default.
                }
            }

            ideal[0] = (this.modelConfig.getPosTags().contains(tag) ? 1.0d : 0.0d);

            int i = 0;
            for(Integer columnId: this.trainingDataSet.getDataColumnIdList()) {
                List<Double> normalizedData = Normalizer.normalize(columnConfigList.get(columnId), fields[columnId]);
                for ( Double normalValue : normalizedData ) {
                    inputs[i++] = normalValue;
                }
            }

            trainingDataSet.addTrainingRecord(new TrainingRecord(inputs, ideal, significance));
        }
    }

    private boolean isPosOrNegTag(ModelConfig config, String tag) {
        return config.getPosTags().contains(tag) || config.getNegTags().contains(tag);
    }

    private List<Integer> getNormalizedColumnIdList() {
        List<Integer> normalizedColumnIdList = new ArrayList<Integer>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig config: columnConfigList) {
            if(CommonUtils.isGoodCandidate(config, hasCandidates)) {
                normalizedColumnIdList.add(config.getColumnNum());
            }
        }
        return normalizedColumnIdList;
    }

    private int getTargetColumnCount() {
        int targetCount = 0;

        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                targetCount++;
            }
        }
        return targetCount;
    }
}
