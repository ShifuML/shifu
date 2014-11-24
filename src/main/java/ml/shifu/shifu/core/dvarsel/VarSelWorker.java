package ml.shifu.shifu.core.dvarsel;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.google.common.base.Splitter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.mapreduce.GuaguaLineRecordReader;
import ml.shifu.guagua.mapreduce.GuaguaWritableAdapter;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingRecord;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created on 11/24/2014.
 */
public class VarSelWorker extends AbstractWorkerComputable<VarSelMasterResult, VarSelWorkerResult, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    private static final Logger LOG = LoggerFactory.getLogger(VarSelWorker.class);

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;

    private AbstractWorkerConductor workerConductor;

    private long count = 0;
    private int inputNodeCount;
    private int outputNodeCount;

    private TrainingDataSet trainingDataSet;

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader());
        this.getRecordReader().initialize(fileSplit);
    }

    @Override
    public void init(WorkerContext<VarSelMasterResult, VarSelWorkerResult> workerContext) {
        Properties props = workerContext.getProps();

        try {
            RawSourceData.SourceType sourceType = RawSourceData.SourceType.valueOf(
                    props.getProperty(NNConstants.NN_MODELSET_SOURCE_TYPE, RawSourceData.SourceType.HDFS.toString()));

            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(NNConstants.SHIFU_NN_MODEL_CONFIG),
                    sourceType);

            this.columnConfigList = CommonUtils.loadColumnConfigList(NNConstants.SHIFU_NN_COLUMN_CONFIG, sourceType);

            String conductorClsName = props.getProperty("dvarsel.worker.conductor.cls");

            this.workerConductor = (AbstractWorkerConductor) Class.forName(conductorClsName)
                    .getDeclaredConstructor(ModelConfig.class, List.class)
                    .newInstance(this.modelConfig, this.columnConfigList);

        } catch ( IOException e ) {
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
    }

    @Override
    public VarSelWorkerResult doCompute(WorkerContext<VarSelMasterResult, VarSelWorkerResult> workerContext) {
        if ( !workerConductor.isInitialized() ) {
            workerConductor.retainData(trainingDataSet);
        }

        VarSelMasterResult masterResult = workerContext.getLastMasterResult();
        if (masterResult == null) {
            // no working set, wait master to send the working set
            return new VarSelWorkerResult(-1);
        }

        if ( masterResult.isHalt() ) {
            // finish variable selection, stop working
            return null;
        }

        workerConductor.consumeMasterResult(masterResult);
        return workerConductor.generateVarSelResult();
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey,
                     GuaguaWritableAdapter<Text> currentValue,
                     WorkerContext<VarSelMasterResult, VarSelWorkerResult> workerContext) {
        if ( (++this.count) % 100000 == 0 ) {
            LOG.info("Read {} records.", this.count);
        }

        double[] inputs = new double[this.inputNodeCount];
        double[] ideal = new double[this.outputNodeCount];
        double significance = NNConstants.DEFAULT_SIGNIFICANCE_VALUE;

        int elementCnt = 0;
        for (String input : Splitter.on(NNConstants.NN_DEFAULT_COLUMN_SEPARATOR)
                .split(currentValue.getWritable().toString())) {
            double dval = NumberFormatUtils.getDouble(input.trim(), 0.0d);

            if ( elementCnt < this.outputNodeCount ) {
                ideal[elementCnt ++] = dval;
            } else {
                int inputsIndex = (elementCnt ++) - this.outputNodeCount;
                if ( inputsIndex < this.inputNodeCount ) {
                    inputs[inputsIndex] = dval;
                } else if ( inputsIndex == this.inputNodeCount ) {
                    significance = dval;
                } else {
                    break;
                }
            }
        }

        if ( elementCnt != this.inputNodeCount + this.outputNodeCount + 1 ) {
            // not enough data
            LOG.warn("Incomplete data... expected field count - {}, but actual - {}",
                    (this.inputNodeCount + this.outputNodeCount + 1), elementCnt);
        } else {
            trainingDataSet.addTrainingRecord(new TrainingRecord(inputs, ideal, significance));
        }
    }

    private List<Integer> getNormalizedColumnIdList() {
        List<Integer> normalizedColumnIdList = new ArrayList<Integer>();
        for (ColumnConfig config : columnConfigList) {
            if (config.isCandidate()) {
                normalizedColumnIdList.add(config.getColumnNum());
            }
        }

        return normalizedColumnIdList;
    }

    private int getTargetColumnCount() {
        int targetCount = 0;

        for (ColumnConfig config : columnConfigList) {
            if (config.isTarget()) {
                targetCount++;
            }
        }
        return targetCount;
    }
}
