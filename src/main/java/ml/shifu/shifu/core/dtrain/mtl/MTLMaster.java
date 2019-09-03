/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.mtl;

import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.*;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * @author haillu
 */
public class MTLMaster extends AbstractMasterComputable<MTLParams, MTLParams> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTLMaster.class);

    private ModelConfig modelConfig;

    private List<List<ColumnConfig>> mtlColumnConfigLists = new ArrayList<>();

    private boolean isContinuousEnabled = false;

    private int inputCount;

    private Map<String, Object> validParams;

    private MultiTaskLearning mtl;
    private int taskNumber;

    @Override
    public void init(MasterContext<MTLParams, MTLParams> context) {
        LOG.info("master init:");
        Properties props = context.getProps();
        SourceType sourceType = SourceType
                .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));

        loadConfigs(props, sourceType);

        for(List<ColumnConfig> ccs: mtlColumnConfigLists) {
            int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(ccs);
            this.inputCount += inputOutputIndex[0] + inputOutputIndex[1];
        }
        this.isContinuousEnabled = Boolean.TRUE.toString()
                .equalsIgnoreCase(props.getProperty(CommonConstants.CONTINUOUS_TRAINING));

        // build multi-task learning model:
        this.validParams = this.modelConfig.getTrain().getParams();
        double learningRate = (double) validParams.get(CommonConstants.LEARNING_RATE);
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        List<String> hiddenActiFuncs = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);

        // todo:check whether MTL need regression function and decide how to add it.
        double l2reg = 0;
        Object pObject = this.validParams.get(CommonConstants.PROPAGATION);
        String propagation = (pObject == null) ? DTrainUtils.RESILIENTPROPAGATION : pObject.toString();

        LOG.info("params of constructor of MTL: inputCount: {},hiddenNodes: {},hiddenActiFuncs: {}"
                + "taskNumber: {},l2reg: {}", inputCount, hiddenNodes, hiddenActiFuncs, taskNumber, l2reg);

        this.mtl = new MultiTaskLearning(inputCount, hiddenNodes, hiddenActiFuncs, taskNumber, l2reg);
        this.mtl.initOptimizer(learningRate, propagation, 0, RegulationLevel.NONE);
    }

    private void loadConfigs(Properties props, SourceType sourceType) {
        try {
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);

            // build mtlColumnConfigLists.
            List<String> tagColumns = this.modelConfig.getMultiTaskTargetColumnNames();
            this.taskNumber = tagColumns.size();
            PathFinder pf = new PathFinder(this.modelConfig);
            for(int i = 0; i < this.taskNumber; i++) {
                List<ColumnConfig> ccs;
                ccs = CommonUtils.loadColumnConfigList(pf.getMTLColumnConfigPath(sourceType, i), sourceType);
                // for local test
                // ccs = CommonUtils.loadColumnConfigList(
                // "/C:/Users/haillu/Documents/gitRepo/shifu/target/test-classes/model/MultiTaskNN/mtl/ColumnConfig.json."+i,
                // sourceType);
                mtlColumnConfigLists.add(ccs);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public MTLParams doCompute(MasterContext<MTLParams, MTLParams> context) {
        if(context.isFirstIteration()) {
            return initOrRecoverModelWeights(context);
        }

        MTLParams aggregation = aggregateWorkerGradients(context);
        this.mtl.optimizeWeight(aggregation.getTrainSize(), context.getCurrentIteration() - 1, aggregation.getMtl());
        MTLParams params = new MTLParams();

        LOG.info(
                "params will be sent to worker: trainSize:{},validationSize:{},"
                        + "trainCount:{},validationCount:{},trainError:{},validationErrors:{}",
                aggregation.getTrainSize(), aggregation.getValidationSize(), aggregation.getTrainCount(),
                aggregation.getValidationCount(), aggregation.getTrainErrors(), aggregation.getValidationErrors());

        params.setTrainSize(aggregation.getTrainSize());
        params.setValidationSize(aggregation.getValidationSize());
        params.setTrainCount(aggregation.getTrainCount());
        params.setValidationCount(aggregation.getValidationCount());
        params.setTrainErrors(aggregation.getTrainErrors());
        params.setValidationErrors(aggregation.getValidationErrors());
        params.setSerializationType(SerializationType.WEIGHTS);
        this.mtl.setSerializationType(SerializationType.WEIGHTS);
        params.setMtl(this.mtl);
        return params;
    }

    public MTLParams aggregateWorkerGradients(MasterContext<MTLParams, MTLParams> context) {
        MTLParams aggregation = null;
        for(MTLParams params: context.getWorkerResults()) {
            if(aggregation == null) {
                aggregation = params;
            } else {
                aggregation.combine(params);
            }
        }
        return aggregation;
    }

    public MTLParams initOrRecoverModelWeights(MasterContext<MTLParams, MTLParams> context) {
        MTLParams params = new MTLParams();
        if(this.isContinuousEnabled) {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            MultiTaskLearning existingModel = loadModel(modelPath);
            if(existingModel != null) {
                this.mtl.updateWeights(existingModel);
            } else {
                LOG.warn("Continuous training enabled but existing model load failed, do random initialization.");
                this.mtl.initWeights();
            }
        } else {
            this.mtl.initWeights();
        }
        params.setMtl(this.mtl);
        return params;
    }

    public MultiTaskLearning loadModel(Path modelPath) {
        FileSystem fileSystem = ShifuFileUtils.getFileSystemBySourceType(SourceType.HDFS);
        InputStream inputStream = null;
        try {
            inputStream = fileSystem.open(modelPath);
            return IndependentMTLModel.loadFromStream(inputStream).getMtl();
        } catch (IOException e) {
            LOG.error("IOException happen when load MultiTaskLearning from HDFS", e);
        } finally {
            IOUtils.closeQuietly(inputStream);
        }
        return null;
    }
}
