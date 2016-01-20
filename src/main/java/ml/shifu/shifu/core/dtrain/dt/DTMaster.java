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
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Queue;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams.NodeStats;
import ml.shifu.shifu.util.CommonUtils;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DTMaster extends AbstractMasterComputable<DTMasterParams, DTWorkerParams> {

    private static final Logger LOG = LoggerFactory.getLogger(DTMaster.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    private int treeNum;

    private List<TreeNode> trees;

    private Queue<TreeNode> queue;

    private FeatureSubsetStrategy featureSubsetStrategy = FeatureSubsetStrategy.ALL;

    private int maxDepth;

    private long maxStatsMemory;

    /**
     * If variables are selected, if not, select variables with good candidate.
     */
    private boolean isAfterVarSelect;

    private Impurity impurity;

    @Override
    public DTMasterParams doCompute(MasterContext<DTMasterParams, DTWorkerParams> context) {
        if(context.isFirstIteration()) {
            return buildInitialMasterParams();
        } else {
            Map<Integer, NodeStats> nodeStatsMap = mergeWorkerResults(context.getWorkerResults());
            LOG.debug("node stats after merged: {}", nodeStatsMap);
            for(Entry<Integer, NodeStats> entry: nodeStatsMap.entrySet()) {
                NodeStats nodeStats = entry.getValue();
                int treeId = nodeStats.getTreeId();
                Node doneNode = Node.getNode(trees.get(treeId).getNode(), nodeStats.getNodeId());
                LOG.info("Node stats with treeId {} and node id {},", treeId, doneNode.getId());
                // doneNode, NodeStats
                Map<Integer, double[]> statistics = nodeStats.getFeatureStatistics();
                List<GainInfo> gainList = new ArrayList<GainInfo>();
                for(Entry<Integer, double[]> gainEntry: statistics.entrySet()) {
                    int columnNum = gainEntry.getKey();
                    ColumnConfig config = this.columnConfigList.get(columnNum);
                    double[] statsArray = gainEntry.getValue();
                    gainList.add(this.impurity.computeImpurity(statsArray, config));
                }

                GainInfo maxGainInfo = GainInfo.getGainInfoByMaxGain(gainList);
                populateGainInfoToNode(doneNode, maxGainInfo);
                LOG.info("GainInfo is {} and node with info is {}.", maxGainInfo, doneNode);

                boolean isLeaf = maxGainInfo.getGain() <= 0 || Node.indexToLevel(doneNode.getId()) == this.maxDepth;
                doneNode.setLeaf(isLeaf);
                if(!doneNode.isLeaf()) {
                    boolean leftChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                            || (maxGainInfo.getLeftImpurity() == 0.0);
                    Node left = new Node(Node.leftIndex(doneNode.getId()), maxGainInfo.getLeftPredict(),
                            maxGainInfo.getLeftImpurity(), leftChildIsLeaf);
                    boolean rightChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                            || (maxGainInfo.getRightImpurity() == 0.0);
                    Node right = new Node(Node.rightIndex(doneNode.getId()), maxGainInfo.getRightPredict(),
                            maxGainInfo.getRightImpurity(), rightChildIsLeaf);
                    doneNode.setLeft(left);
                    if(!leftChildIsLeaf) {
                        this.queue.offer(new TreeNode(treeId, left));
                    }

                    doneNode.setRight(right);
                    if(!rightChildIsLeaf) {
                        this.queue.offer(new TreeNode(treeId, right));
                    }
                }
            }

            Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>();
            DTMasterParams masterParams = new DTMasterParams();
            if(queue.isEmpty()) {
                masterParams.setHalt(true);
                LOG.info("Queue is empty, training is stopped in iteration {}.", context.getCurrentIteration());
            } else {
                int nodeIndexInGroup = 0;
                long currMem = 0L;
                while(!queue.isEmpty() && currMem <= this.maxStatsMemory) {
                    TreeNode node = this.queue.poll();
                    List<Integer> subsetFeatures = getSubsamplingFeatures(featureSubsetStrategy);
                    node.setFeatures(subsetFeatures);
                    currMem += getStatsMem(subsetFeatures);
                    todoNodes.put(nodeIndexInGroup, node);
                    nodeIndexInGroup += 1;
                }
                masterParams.setTodoNodes(todoNodes);
                LOG.info("Todo nodes with size {}.", todoNodes.size());
            }
            masterParams.setTrees(trees);
            return masterParams;
        }
    }

    private void populateGainInfoToNode(Node doneNode, GainInfo maxGainInfo) {
        doneNode.setPredict(maxGainInfo.getPredict());
        doneNode.setSplit(maxGainInfo.getSplit());
        doneNode.setGain(maxGainInfo.getGain());
        doneNode.setImpurity(maxGainInfo.getImpurity());
        doneNode.setLeftImpurity(maxGainInfo.getLeftImpurity());
        doneNode.setRightImpurity(maxGainInfo.getRightImpurity());
        doneNode.setLeftPredict(maxGainInfo.getLeftPredict());
        doneNode.setRightPredict(maxGainInfo.getRightPredict());
    }

    private long getStatsMem(List<Integer> subsetFeatures) {
        long statsMem = 0L;
        for(Integer columnNum: subsetFeatures) {
            ColumnConfig config = this.columnConfigList.get(columnNum);
            // 1.5 is overhead for java object
            if(config.isNumerical()) {
                statsMem += (config.getBinBoundary().size() + 1) * this.impurity.getStatsSize() * 8L * 1.5;
            } else if(config.isCategorical()) {
                statsMem += (config.getBinCategory().size() + 1) * this.impurity.getStatsSize() * 8L * 1.5;
            }
        }
        return statsMem;
    }

    private Map<Integer, NodeStats> mergeWorkerResults(Iterable<DTWorkerParams> workerResults) {
        boolean isFirst = false;
        Map<Integer, NodeStats> nodeStatsMap = null;
        for(DTWorkerParams params: workerResults) {
            if(!isFirst) {
                isFirst = true;
                nodeStatsMap = params.getNodeStatsMap();
            } else {
                Map<Integer, NodeStats> currNodeStatsmap = params.getNodeStatsMap();
                for(Entry<Integer, NodeStats> entry: nodeStatsMap.entrySet()) {
                    NodeStats resultNodeStats = entry.getValue();
                    mergeNodeStats(resultNodeStats, currNodeStatsmap.get(entry.getKey()));
                }
            }
        }
        return nodeStatsMap;
    }

    private void mergeNodeStats(NodeStats resultNodeStats, NodeStats nodeStats) {
        Map<Integer, double[]> featureStatistics = resultNodeStats.getFeatureStatistics();
        for(Entry<Integer, double[]> entry: nodeStats.getFeatureStatistics().entrySet()) {
            double[] statistics = featureStatistics.get(entry.getKey());
            for(int i = 0; i < statistics.length; i++) {
                statistics[i] += entry.getValue()[i];
            }
        }
    }

    private DTMasterParams buildInitialMasterParams() {
        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>();
        int nodeIndexInGroup = 0;
        for(TreeNode treeNode: trees) {
            List<Integer> features = getSubsamplingFeatures(this.featureSubsetStrategy);
            treeNode.setFeatures(features);
            todoNodes.put(nodeIndexInGroup, treeNode);
            nodeIndexInGroup += 1;
        }
        return new DTMasterParams(trees, todoNodes);
    }

    private List<Integer> getSubsamplingFeatures(FeatureSubsetStrategy featureSubsetStrategy) {
        List<Integer> features = new ArrayList<Integer>();
        Random random = new Random();
        switch(featureSubsetStrategy) {
            case HALF:
                for(ColumnConfig config: columnConfigList) {
                    if(isAfterVarSelect) {
                        if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                            if(random.nextDouble() >= 0.5d) {
                                features.add(config.getColumnNum());
                            }
                        }
                    } else {
                        if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                            if(random.nextDouble() >= 0.5d) {
                                features.add(config.getColumnNum());
                            }
                        }
                    }
                }
                break;
            case ONETHIRD:
                for(ColumnConfig config: columnConfigList) {
                    if(isAfterVarSelect) {
                        if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                            if(random.nextDouble() < 0.33d) {
                                features.add(config.getColumnNum());
                            }
                        }
                    } else {
                        if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                            if(random.nextDouble() < 0.33d) {
                                features.add(config.getColumnNum());
                            }
                        }
                    }
                }
                break;
            case ALL:
            default:
                // an empty list means all
                break;
        }
        return features;
    }

    @Override
    public void init(MasterContext<DTMasterParams, DTWorkerParams> context) {
        Properties props = context.getProps();
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;

        this.featureSubsetStrategy = FeatureSubsetStrategy.of(this.modelConfig.getTrain().getParams()
                .get("featureSubsetStrategy").toString());

        this.maxDepth = Integer.valueOf(this.modelConfig.getTrain().getParams().get("maxDepth").toString());
        assert this.maxDepth > 0 && this.maxDepth <= 20;

        this.maxStatsMemory = Long.valueOf(this.modelConfig.getTrain().getParams().get("maxStatsMemoryMB").toString()) * 1024 * 1024;
        assert this.maxStatsMemory <= Math.min(Runtime.getRuntime().maxMemory() * 0.6, 800 * 1024 * 1024L);

        this.treeNum = this.modelConfig.getTrain().getBaggingNum();

        String imStr = this.modelConfig.getTrain().getParams().get("impurity").toString();
        int numClasses = 2;
        if(this.modelConfig.isMultiClassification()) {
            numClasses = this.modelConfig.getFlattenTags().size();
        }
        if(imStr.equalsIgnoreCase("entropy")) {
            impurity = new Entropy(numClasses);
        } else if(imStr.equalsIgnoreCase("gini")) {
            impurity = new Gini(numClasses);
        } else {
            impurity = new Variance();
        }
        LOG.info("Master init params: isAfterVarSel={}, featureSubsetStrategy={}, maxDepth={}, maxStatsMemory={}, "
                + "treeNum={}, impurity= {}", isAfterVarSelect, featureSubsetStrategy, maxDepth, maxStatsMemory,
                treeNum, imStr);
        this.trees = new ArrayList<TreeNode>(treeNum);
        for(int i = 0; i < treeNum; i++) {
            this.trees.add(new TreeNode(i, new Node(Node.ROOT_INDEX)));
        }

        this.queue = new LinkedList<TreeNode>();
        // TODO recover state trees and queue here for fail-over
    }
}
