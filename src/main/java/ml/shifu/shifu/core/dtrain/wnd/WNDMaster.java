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
package ml.shifu.shifu.core.dtrain.wnd;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

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
import ml.shifu.shifu.util.CommonUtils;

/**
 * TODO master aggregation logic to aggregate gradients and update weights based on different optimization strategies
 * like ADAM, AdaGrad, SGD ...
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WNDMaster extends AbstractMasterComputable<WNDParams, WNDParams> {

    protected static final Logger LOG = LoggerFactory.getLogger(WNDMaster.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * If variables are selected, if not, select variables with good candidate.
     */
    private boolean isAfterVarSelect;

    /**
     * Learning rate
     */
    @SuppressWarnings("unused")
    private double learningRate;

    /**
     * Whether to enable continuous model training based on existing models.
     */
    @SuppressWarnings("unused")
    private boolean isContinuousEnabled = false;

    /**
     * Every checkpoint interval, do checkpoint to save {@link #trees} and {@link #toDoQueue} and MasterParams in that
     * iteration.
     */
    @SuppressWarnings("unused")
    private int checkpointInterval;

    /**
     * Checkpoint output HDFS file
     */
    @SuppressWarnings("unused")
    private Path checkpointOutput;

    private int numInputs;


    private Map<String, Object> validParams;

    private WideAndDeep wnd;

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.master.AbstractMasterComputable#init(ml.shifu.guagua.master.MasterContext)
     */
    @SuppressWarnings({ "unchecked", "unused" })
    @Override
    public void init(MasterContext<WNDParams, WNDParams> context) {
        Properties props = context.getProps();
        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        // numerical + categorical = # of all input
        this.numInputs = inputOutputIndex[0];
        // regression outputNodeCount is 1, binaryClassfication, it is 1, OneVsAll it is 1, Native classification it is
        // 1, with index of 0,1,2,3 denotes different classes
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        this.validParams = this.modelConfig.getTrain().getParams();
        this.learningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());

        // Build wide and deep graph
        List<Integer> embedColumnIds = (List<Integer>) this.validParams.get(CommonConstants.NUM_EMBED_COLUMN_IDS);
        Integer embedOutputs = (Integer) this.validParams.get(CommonConstants.NUM_EMBED_OUTPUTS);
        List<Integer> embedOutputList = new ArrayList<Integer>();
        for(Integer cId: embedColumnIds) {
            embedOutputList.add(embedOutputs == null ? 16 : embedOutputs);
        }
        List<Integer> wideColumnIds = DTrainUtils.getCategoricalIds(columnConfigList, isAfterVarSelect);
        int numLayers = (Integer) this.validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        Float l2reg = (Float) this.validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        this.wnd = new WideAndDeep(columnConfigList, numInputs, embedColumnIds, embedOutputList, wideColumnIds,
                hiddenNodes, actFunc, l2reg);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.master.AbstractMasterComputable#doCompute(ml.shifu.guagua.master.MasterContext)
     */
    @Override
    public WNDParams doCompute(MasterContext<WNDParams, WNDParams> context) {
        if(context.isFirstIteration()) {
            WNDParams params = new WNDParams(); // TODO, init weights in WideAndDeep with this.wnd object
            params.setWnd(this.wnd); //weights from this.wnd
            return params;
        }
        
        WNDParams aggregation = null;
        for(WNDParams params: context.getWorkerResults()) {
            if(aggregation == null) {
                aggregation = params;
            } else {
                aggregation.combine(params);
            }
        }
        
        // TODO optimizer, wnd object as current model weights, aggregation as current iteration gradients aggregation
        // gradients = aggregation.getWnd();
        // this.wnd -= this.learningRate * gradients;
        
        // construct master result which contains WideAndDeep current model weights
        WNDParams params = new WNDParams();
        params.setTrainCount(aggregation.getTrainCount());
        params.setValidationCount(aggregation.getValidationCount());
        params.setTrainError(aggregation.getTrainError());
        params.setValidationError(aggregation.getValidationError());
        params.setWnd(this.wnd);
        return params;
    }

}
