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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Queue;
import java.util.Random;

import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams.NodeStats;
import ml.shifu.shifu.util.CommonUtils;

/**
 * Random forest and gradient boost decision tree {@link MasterComputable} implementation.
 * 
 * <p>
 * {@link #isRF} and {@link #isGBDT} are for RF or GBDT checking, by default RF is trained.
 * 
 * <p>
 * Each iteration, update node statistics and determine best split which is used for tree node split. Besides node
 * statistics, error and count info are also collected for client display.
 * 
 * <p>
 * Each iteration, new node group with nodes in limited estimated memory consumption are sent out to all workers for
 * feature statistics.
 * 
 * <p>
 * For gradient boost decision tree, each time a tree is updated and if one tree is finalized, then start a new tree.
 * Both random forest and gradient boost decision trees are all stored in {@link #trees}.
 * 
 * <p>
 * Terminal condition: for random forest, just to collect all nodes in all trees from all workers. Terminal condition is
 * all trees cannot be split. If one tree cannot be split with threshold count and meaningful impurity, one tree if
 * finalized and stopped update. For gradient boost decision tree, each time only one tree is trained, if last tree
 * cannot be split which means training is stopped.
 * 
 * <p>
 * TODO In current {@link DTMaster}, there are states like {@link #trees} and {@link #queue} and cannot be updated.
 * Consider only one {@link MasterComputable} instance in one Guagua job. The down rate of such master small. To
 * consider master fail over, such states should all be recovered in {@link #init(MasterContext)}.
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

    /**
     * Number of trees for both RF and GBDT
     */
    private int treeNum;

    /**
     * Feature sub sampling strategy: ALL, HALF, ONETHIRD
     */
    private FeatureSubsetStrategy featureSubsetStrategy = FeatureSubsetStrategy.ALL;

    /**
     * Max depth of a tree, by default is 10
     */
    private int maxDepth;

    /**
     * Max stats memory to group nodes.
     */
    private long maxStatsMemory;

    /**
     * If variables are selected, if not, select variables with good candidate.
     */
    private boolean isAfterVarSelect;

    /**
     * Different {@link Impurity} for node, {@link Entropy} and {@link Gini} are mostly for classification,
     * {@link Variance} are mostly for regression.
     */
    private Impurity impurity;

    /**
     * If for random forest running, this is default for such master.
     */
    private boolean isRF = true;

    /**
     * If gradient boost decision tree, for GBDT, each time a tree is trained, next train is trained by gradient label
     * from previous tree.
     */
    private boolean isGBDT = false;

    /**
     * Learning rate for GBDT.
     */
    @SuppressWarnings("unused")
    private double learningRate;

    // ############################################################################################################
    // ## There parts are states, for fail over such instances should be recovered in {@link #init(MasterContext)}
    // ############################################################################################################

    /**
     * All trees trained in this master
     */
    private List<TreeNode> trees;

    /**
     * TreeNodes needed to be collected statistics from workers.
     */
    private Queue<TreeNode> queue;

    @Override
    public DTMasterParams doCompute(MasterContext<DTMasterParams, DTWorkerParams> context) {
        if(context.isFirstIteration()) {
            return buildInitialMasterParams();
        }
        boolean isFirst = false;
        Map<Integer, NodeStats> nodeStatsMap = null;
        double squareError = 0d;
        long count = 0L;
        for(DTWorkerParams params: context.getWorkerResults()) {
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
            squareError += params.getSquareError();
            if(this.isRF) {
                count += params.getCount() * this.treeNum;
            }
            if(this.isGBDT) {
                count += params.getCount();
            }
        }
        LOG.debug("node stats after merged: {}", nodeStatsMap);
        for(Entry<Integer, NodeStats> entry: nodeStatsMap.entrySet()) {
            NodeStats nodeStats = entry.getValue();
            int treeId = nodeStats.getTreeId();
            Node doneNode = Node.getNode(trees.get(treeId).getNode(), nodeStats.getNodeId());
            LOG.debug("Node stats with treeId {} and node id {},", treeId, doneNode.getId());
            // doneNode, NodeStats
            Map<Integer, double[]> statistics = nodeStats.getFeatureStatistics();
            LOG.debug("Node stats {},", statistics);

            List<GainInfo> gainList = new ArrayList<GainInfo>();
            for(Entry<Integer, double[]> gainEntry: statistics.entrySet()) {
                int columnNum = gainEntry.getKey();
                ColumnConfig config = this.columnConfigList.get(columnNum);
                double[] statsArray = gainEntry.getValue();
                // LOG.info("fstatssss: {}", Arrays.toString(statsArray));
                gainList.add(this.impurity.computeImpurity(statsArray, config));
            }

            GainInfo maxGainInfo = GainInfo.getGainInfoByMaxGain(gainList);
            populateGainInfoToNode(doneNode, maxGainInfo);
            LOG.info("GainInfo is {} and node with info is {}.", maxGainInfo, doneNode);

            // another stop condition: max instandce count
            boolean isLeaf = maxGainInfo.getGain() <= 0d || Node.indexToLevel(doneNode.getId()) == this.maxDepth;
            doneNode.setLeaf(isLeaf);
            if(!doneNode.isLeaf()) {
                boolean leftChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                        || (maxGainInfo.getLeftImpurity() == 0.0);
                // such node is just set into isLeaf to true
                int leftIndex = Node.leftIndex(doneNode.getId());
                Node left = new Node(leftIndex, maxGainInfo.getLeftPredict(), maxGainInfo.getLeftImpurity(), true);
                doneNode.setLeft(left);
                if(!leftChildIsLeaf) {
                    this.queue.offer(new TreeNode(treeId, left));
                }

                boolean rightChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                        || (maxGainInfo.getRightImpurity() == 0.0);
                // such node is just set into isLeaf to true
                int rightIndex = Node.rightIndex(doneNode.getId());
                Node right = new Node(rightIndex, maxGainInfo.getRightPredict(), maxGainInfo.getRightImpurity(), true);
                doneNode.setRight(right);
                if(!rightChildIsLeaf) {
                    this.queue.offer(new TreeNode(treeId, right));
                }
            }
        }

        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>();
        DTMasterParams masterParams = new DTMasterParams(count, squareError);
        if(queue.isEmpty()) {
            if(this.isGBDT) {
                if(this.trees.size() == this.treeNum) {
                    masterParams.setHalt(true);
                    LOG.info("Queue is empty, training is stopped in iteration {}.", context.getCurrentIteration());
                } else {
                    TreeNode newRootNode = new TreeNode(this.trees.size(), new Node(Node.ROOT_INDEX));
                    this.trees.add(newRootNode);
                    newRootNode.setFeatures(getSubsamplingFeatures(this.featureSubsetStrategy));
                    // only one node
                    todoNodes.put(0, newRootNode);
                    masterParams.setTodoNodes(todoNodes);
                    // set switch flag
                    masterParams.setSwitchToNextTree(true);
                }
            } else {
                masterParams.setHalt(true);
                LOG.info("Queue is empty, training is stopped in iteration {}.", context.getCurrentIteration());
            }
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
                statsMem += config.getBinBoundary().size() * this.impurity.getStatsSize() * 8L * 1.5;
            } else if(config.isCategorical()) {
                statsMem += (config.getBinCategory().size() + 1) * this.impurity.getStatsSize() * 8L * 1.5;
            }
        }
        return statsMem;
    }

    private void mergeNodeStats(NodeStats resultNodeStats, NodeStats nodeStats) {
        Map<Integer, double[]> featureStatistics = resultNodeStats.getFeatureStatistics();
        for(Entry<Integer, double[]> entry: nodeStats.getFeatureStatistics().entrySet()) {
            double[] statistics = featureStatistics.get(entry.getKey());
            for(int i = 0; i < statistics.length; i++) {
                statistics[i] += entry.getValue()[i];
            }
            LOG.debug("statistics {}", Arrays.toString(statistics));
        }
    }

    private DTMasterParams buildInitialMasterParams() {
        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>(treeNum, 1.0f);
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

        // init model config and column config list at first
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

        // check if variables are set final selected
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;

        // tree related parameters initialization
        this.featureSubsetStrategy = FeatureSubsetStrategy.of(this.modelConfig.getTrain().getParams()
                .get("FeatureSubsetStrategy").toString());
        this.maxDepth = Integer.valueOf(this.modelConfig.getTrain().getParams().get("MaxDepth").toString());
        assert this.maxDepth > 0 && this.maxDepth <= 20;
        this.maxStatsMemory = Long.valueOf(this.modelConfig.getTrain().getParams().get("MaxStatsMemoryMB").toString()) * 1024 * 1024;
        assert this.maxStatsMemory <= Math.min(Runtime.getRuntime().maxMemory() * 0.6, 800 * 1024 * 1024L);
        this.treeNum = this.modelConfig.getTrain().getBaggingNum();
        this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        this.isGBDT = ALGORITHM.GBDT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        if(this.isGBDT) {
            // learning rate only effective in gbdt
            this.learningRate = Double.valueOf(this.modelConfig.getParams().get(NNTrainer.LEARNING_RATE).toString());
        }
        String imStr = this.modelConfig.getTrain().getParams().get("Impurity").toString();
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
        this.queue = new LinkedList<TreeNode>();

        // initialize trees
        if(context.isFirstIteration()) {
            if(this.isRF) {
                // for random forest, trees are trained in parallel
                this.trees = new ArrayList<TreeNode>(treeNum);
                for(int i = 0; i < treeNum; i++) {
                    this.trees.add(new TreeNode(i, new Node(Node.ROOT_INDEX)));
                }
            }
            if(this.isGBDT) {
                this.trees = new ArrayList<TreeNode>(treeNum);
                // for GBDT, initialize the first tree. trees are trained sequentially
                this.trees.add(new TreeNode(0, new Node(Node.ROOT_INDEX)));
            }
        } else {
            this.trees = context.getMasterResult().getTrees();
            // TODO recover state trees and queue here for fail-over
        }
    }
}
