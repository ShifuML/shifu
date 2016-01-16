/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.dt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.util.CommonUtils;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DTMaster implements MasterComputable<DTMasterParams, DTWorkerParams> {

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Whether some configurations are initialized
     */
    private AtomicBoolean isInitialized = new AtomicBoolean(false);

    private int treeNum;

    private List<TreeNode> trees;

    @SuppressWarnings("unused")
    private FeatureSubsetStrategy featureSubsetStrategy = FeatureSubsetStrategy.ALL;

    @Override
    public DTMasterParams compute(MasterContext<DTMasterParams, DTWorkerParams> context) {
        if(isInitialized.compareAndSet(false, true)) {
            init(context);
            return buildInitialMasterParams();
        }
        
        // TODO oters 
        return null;
    }

    private DTMasterParams buildInitialMasterParams() {
        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>();
        for(TreeNode treeNode: trees) {
            treeNode.setFeatures(getSubsamplingFeatures());
            todoNodes.put(treeNode.getTreeIndex(), treeNode);
        }
        return new DTMasterParams(trees, todoNodes);
    }

    private List<Integer> getSubsamplingFeatures() {
        // TODO according to featureSubsetStrategy
        List<Integer> features = new ArrayList<Integer>();
        for(ColumnConfig config: columnConfigList) {
            features.add(config.getColumnNum());
        }
        return features;
    }

    private void init(MasterContext<DTMasterParams, DTWorkerParams> context) {
        Properties props = context.getProps();
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(NNConstants.NN_MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(NNConstants.SHIFU_NN_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(NNConstants.SHIFU_NN_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.treeNum = this.modelConfig.getTrain().getBaggingNum();
        this.trees = new ArrayList<TreeNode>(treeNum);
        for(int i = 0; i < treeNum; i++) {
            // Node index start from 1, set to 1 means root node
            this.trees.add(new TreeNode(i, new Node(1)));
        }
    }

}
