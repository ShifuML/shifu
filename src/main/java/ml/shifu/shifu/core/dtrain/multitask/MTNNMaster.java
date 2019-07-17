package ml.shifu.shifu.core.dtrain.multitask;


import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;
import ml.shifu.shifu.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * @author haillu
 * @date 7/17/2019 5:00 PM
 */
public class MTNNMaster extends AbstractMasterComputable<MTNNParams, MTNNParams> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTNNMaster.class);

    private ModelConfig modelConfig;

    private List<ColumnConfig> columnConfigList;

    private boolean isAfterVarSelect;

    private double learningRate;

    private boolean isContinuousEnabled = false;

    private int numInputs;

    private Map<String, Object> validParams;

    private MultiTaskNN multiTaskNN;

    private Optimizer optimizer;


    @Override
    public void init(MasterContext<MTNNParams, MTNNParams> context) {
        Properties props = context.getProps();
        SourceType sourceType = SourceType.
                valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
        try {
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException();
        }

    }

    @Override
    public MTNNParams doCompute(MasterContext<MTNNParams, MTNNParams> context) {
        return null;
    }
}
