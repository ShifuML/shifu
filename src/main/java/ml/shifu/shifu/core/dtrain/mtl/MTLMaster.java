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

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;

/**
 * {@link MTLMaster} is master role in master-workers iterative computing for multi-task model distributed training.
 * 
 * <p>
 * In the first epoch, model weights are random initialized or recovered from HDFS model set path.
 * 
 * <p>
 * Later epochs, worker gradients are accumulated together and model weights in model instance {@link #mtm} are updated
 * based on optimizer configured inside of such master, the updated model weights would be returned and sent out to
 * workers for next epoch computation.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MTLMaster extends AbstractMasterComputable<MTLParams, MTLParams> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTLMaster.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration list loaded from multiple json column configuration files.
     */
    protected List<List<ColumnConfig>> mtlColumnConfigLists;

    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled = false;

    /**
     * # of numerical inputs
     */
    private int numInputs;

    /**
     * Valid parameters from ModelConfig#train#params
     */
    private Map<String, Object> validParams;

    /**
     * Multi-task global model saved into master. Each epoch weights of such model instance would be updated and send
     * back to workers.
     */
    private MultiTaskModel mtm;

    @SuppressWarnings("unchecked")
    @Override
    public void init(MasterContext<MTLParams, MTLParams> context) {
        Properties props = context.getProps();
        loadConfigs(props);

        this.numInputs = 0;
        for(List<ColumnConfig> columnConfigList: mtlColumnConfigLists) {
            int[] inputsAndOutpus = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
            this.numInputs += inputsAndOutpus[0] + inputsAndOutpus[1];
        }

        this.validParams = this.modelConfig.getTrain().getParams();

        this.isContinuousEnabled = Boolean.TRUE.toString()
                .equalsIgnoreCase(context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));

        // Build multiple task model architecture
        List<String> actiFuncs = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        double l2reg = NumberUtils.toDouble(
                this.validParams.getOrDefault(CommonConstants.L2_REG, "").toString(), 0d);
        List<Integer> finalOutputs = new ArrayList<>();
        int tasks = this.modelConfig.getMultiTaskTargetColumnNames().size();
        for(int i = 0; i < tasks; i++) {
            finalOutputs.add(1);
        }
        this.mtm = new MultiTaskModel(this.numInputs, hiddenNodes, actiFuncs, finalOutputs, l2reg);

        // Init multiple task model optimizer
        double learningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());
        Object pObject = this.validParams.get(CommonConstants.PROPAGATION);
        String propagation = (pObject == null) ? DTrainUtils.RESILIENTPROPAGATION : pObject.toString();
        // l2 hard code to NONE here because already set in MultiTaskModel backward
        this.mtm.initOptimizer(learningRate, propagation, 0, RegulationLevel.NONE);
    }

    private void loadConfigs(Properties props) {
        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);

            List<String> tagColumns = this.modelConfig.getMultiTaskTargetColumnNames();
            this.mtlColumnConfigLists = new ArrayList<>();
            PathFinder pathFinder = new PathFinder(this.modelConfig);
            for(int i = 0; i < tagColumns.size(); i++) {
                String ccPath = pathFinder.getMTLColumnConfigPath(sourceType, i);
                this.mtlColumnConfigLists.add(CommonUtils.loadColumnConfigList(ccPath, sourceType));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Master computation logic: 1. first epoch initialize or recover model weights and send to workers for model
     * sync-up; 2. Other epochs, accumulate worker gradients and update global model weights in this master instance.
     */
    @Override
    public MTLParams doCompute(MasterContext<MTLParams, MTLParams> context) {
        if(context.isFirstIteration()) {
            // Fist iteration, no need take worker results and after master model initialization, global model should be
            // sent to workers for training.
            return initOrRecoverModelWeights(context);
        }

        // aggregate all worker gradients to one gradient object.
        MTLParams aggregation = aggregateWorkerGradients(context);

        // apply optimizer
        this.mtm.optimizeWeight(aggregation.getTrainSize(), context.getCurrentIteration() - 1, aggregation.getMtm());

        // construct master result which contains MultiTaskModel current model weights
        MTLParams params = new MTLParams();
        params.setTrainCount(aggregation.getTrainCount());
        params.setValidationCount(aggregation.getValidationCount());
        params.setTrainError(aggregation.getTrainError());
        params.setValidationError(aggregation.getValidationError());
        params.setTrainSize(aggregation.getTrainSize());
        params.setValidationSize(aggregation.getValidationSize());
        params.setSerializationType(SerializationType.WEIGHTS);
        this.mtm.setSerializationType(SerializationType.WEIGHTS);
        params.setMtm(this.mtm);
        return params;
    }

    /**
     * Aggregate worker gradients together into one.
     */
    private MTLParams aggregateWorkerGradients(MasterContext<MTLParams, MTLParams> context) {
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

    /**
     * Random initializing model or recovered from snapshot model from HDFS file for continuous training.
     * 
     * @param context
     *            the work flow context
     * @return the initial MTL parameter
     */
    private MTLParams initOrRecoverModelWeights(MasterContext<MTLParams, MTLParams> context) {
        MTLParams params = new MTLParams();
        if(this.isContinuousEnabled) {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            MultiTaskModel existingModel = loadModel(modelPath);
            if(existingModel != null) {
                this.mtm.updateWeights(existingModel);
            } else {
                LOG.warn("Continuous training enabled but existing model load failed, do random initialization.");
                this.mtm.initWeights();
            }
        } else {
            this.mtm.initWeights();
        }
        params.setMtm(this.mtm); // weights from this.mtm
        return params;
    }

    /**
     * Load snapshot model from hdfs model path.
     * 
     * @param modelPath
     *            hdfs model path
     * @return {@link MultiTaskModel} instance.
     */
    private MultiTaskModel loadModel(Path modelPath) {
        FileSystem fileSystem = ShifuFileUtils.getFileSystemBySourceType(SourceType.HDFS, modelPath);
        InputStream inputStream = null;
        try {
            inputStream = fileSystem.open(modelPath);
            return IndependentMTLModel.loadFromStream(inputStream).getMtm();
        } catch (IOException e) {
            LOG.error("IOException happen when load MultiTaskModel from HDFS", e);
        } finally {
            IOUtils.closeQuietly(inputStream);
        }
        return null;
    }

}
