package ml.shifu.shifu.core.dtrain.multitask;


import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * @author haillu
 */
public class MTNNMaster extends AbstractMasterComputable<MTNNParams, MTNNParams> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTNNMaster.class);

    private ModelConfig modelConfig;

    private List<ColumnConfig> columnConfigList;

    private boolean isAfterVarSelect;

    private double learningRate;

    private boolean isContinuousEnabled = false;

    private int inputCount;

    private Map<String, Object> validParams;

    private MultiTaskNN mtnn;


    @Override
    public void init(MasterContext<MTNNParams, MTNNParams> context) {
        LOG.debug("master init:");
        Properties props = context.getProps();
        SourceType sourceType = SourceType.
                valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
        try {
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException();
        }

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        this.isContinuousEnabled = Boolean.TRUE.toString().equalsIgnoreCase(props.getProperty(CommonConstants.CONTINUOUS_TRAINING));

        //build multi-task nn model:
        this.validParams = this.modelConfig.getTrain().getParams();
        double learningRate = (double) validParams.get(CommonConstants.LEARNING_RATE);
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        List<String> hiddenActiFuncs = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);

        //we have counted it in DTrainUtils.getNumericAndCategoricalInputAndOutputCounts.
        int taskNumber = inputOutputIndex[2];
//        for (ColumnConfig cConfig : this.columnConfigList) {
//            ColumnConfig.ColumnFlag flag = ColumnConfig.ColumnFlag.Target;
//            if (cConfig.getColumnFlag().equals(flag)) {
//                taskNumber++;
//            }
//        }
        // todo:check if MTNN need regression function
        //double l2reg = NumberUtils.toDouble(this.validParams.get(CommonConstants.WDL_L2_REG).toString(), 0);
        double l2reg = 0;
        Object pObject = this.validParams.get(CommonConstants.PROPAGATION);
        String propagation = (pObject == null) ? DTrainUtils.RESILIENTPROPAGATION : pObject.toString();

        LOG.debug("params of constructor of MTNN: inputCount: {},hiddenNodes: {},hiddenActiFuncs: {}" +
                "taskNumber: {},l2reg: {}", inputCount, hiddenNodes, hiddenActiFuncs, taskNumber, l2reg);

        this.mtnn = new MultiTaskNN(inputCount, hiddenNodes, hiddenActiFuncs, taskNumber, l2reg);
        this.mtnn.initOptimizer(learningRate, propagation, 0, RegulationLevel.NONE);
    }

    @Override
    public MTNNParams doCompute(MasterContext<MTNNParams, MTNNParams> context) {
        if (context.isFirstIteration()) {
            return initOrRecoverModelWeights(context);
        }

        MTNNParams aggregation = aggregateWorkerGradients(context);
        this.mtnn.optimizeWeight(aggregation.getTrainSize(), context.getCurrentIteration() - 1, aggregation.getMtnn());
        MTNNParams params = new MTNNParams();

        LOG.debug("params will be sent to worker: trainSize:{},validationSize:{}," +
                        "trainCount:{},validationCount:{},trainError:{},validationErrors:{}", aggregation.getTrainSize(),
                aggregation.getValidationSize(), aggregation.getTrainCount(), aggregation.getValidationCount(),
                aggregation.getTrainErrors(), aggregation.getValidationErrors());

        params.setTrainSize(aggregation.getTrainSize());
        params.setValidationSize(aggregation.getValidationSize());
        params.setTrainCount(aggregation.getTrainCount());
        params.setValidationCount(aggregation.getValidationCount());
        params.setTrainErrors(aggregation.getTrainErrors());
        params.setValidationErrors(aggregation.getValidationErrors());
        params.setSerializationType(SerializationType.WEIGHTS);
        this.mtnn.setSerializationType(SerializationType.WEIGHTS);
        params.setMtnn(this.mtnn);
        return params;
    }

    public MTNNParams aggregateWorkerGradients(MasterContext<MTNNParams, MTNNParams> context) {
        MTNNParams aggregation = null;
        for (MTNNParams params : context.getWorkerResults()) {
            if (aggregation == null) {
                aggregation = params;
            } else {
                aggregation.combine(params);
            }
        }
        return aggregation;
    }

    public MTNNParams initOrRecoverModelWeights(MasterContext<MTNNParams, MTNNParams> context) {
        MTNNParams params = new MTNNParams();
        if (this.isContinuousEnabled) {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            MultiTaskNN existingModel = loadModel(modelPath);
            if (existingModel != null) {
                this.mtnn.updateWeights(existingModel);
            } else {
                LOG.warn("Continuous training enabled but existing model load failed, do random initialization.");
                this.mtnn.initWeights();
            }
        } else {
            this.mtnn.initWeights();
        }
        params.setMtnn(this.mtnn);
        return params;
    }

    public MultiTaskNN loadModel(Path modelPath) {
//        FileSystem fileSystem = ShifuFileUtils.getFileSystemBySourceType(SourceType.HDFS);
//        InputStream inputStream = null;
//        try {
//            inputStream = fileSystem.open(modelPath);
//            return
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        return null;
    }
}
