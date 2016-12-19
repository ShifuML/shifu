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

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams.NodeStats;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    /**
     * How many workers, this is used for memeory usage
     */
    private int workerNumber;

    /**
     * Input features numbers
     */
    private int inputNum;

    /**
     * Cache all features
     */
    private List<Integer> allFeatures;

    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled = false;

    /**
     * If continuous model training, update this to existing tree size, by default is 0, no any impact on existing
     * process.
     */
    private int existingTreeSize = 0;

    /**
     * Every checkpoint interval, do checkpoint to save {@link #trees} and {@link #queue} and MasterParams in that
     * iteration.
     */
    private int checkpointInterval;

    /**
     * Checkpoint output HDFS file
     */
    private Path checkpointOutput;

    /**
     * Common conf to avoid new Configuration
     */
    private Configuration conf;

    /**
     * Checkpoint master params, if only recover queue in fail over, some todo nodes in master result will be ignored.
     * This is used to recover whole stats of that iteration. In {@link #doCompute(MasterContext)}, check if
     * {@link #cpMasterParams} is null, if not, directly return this one and send {@link #cpMasterParams} to null to
     * avoid next iteration to send it again.
     */
    private DTMasterParams cpMasterParams;

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

        if(this.cpMasterParams != null) {
            DTMasterParams tmpMasterParams = rebuildRecoverMasterResultDepthList();
            // set it to null to avoid send it in next iteration
            this.cpMasterParams = null;
            return tmpMasterParams;
        }

        boolean isFirst = false;
        Map<Integer, NodeStats> nodeStatsMap = null;
        double trainError = 0d, validationError = 0d;
        double weightedTrainCount = 0d, weightedValidationCount = 0d;
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
            trainError += params.getTrainError();
            validationError += params.getValidationError();
            weightedTrainCount += params.getTrainCount();
            weightedValidationCount += params.getValidationCount();
        }
        for(Entry<Integer, NodeStats> entry: nodeStatsMap.entrySet()) {
            NodeStats nodeStats = entry.getValue();
            int treeId = nodeStats.getTreeId();
            Node doneNode = Node.getNode(trees.get(treeId).getNode(), nodeStats.getNodeId());
            // doneNode, NodeStats
            Map<Integer, double[]> statistics = nodeStats.getFeatureStatistics();

            List<GainInfo> gainList = new ArrayList<GainInfo>();
            for(Entry<Integer, double[]> gainEntry: statistics.entrySet()) {
                int columnNum = gainEntry.getKey();
                ColumnConfig config = this.columnConfigList.get(columnNum);
                double[] statsArray = gainEntry.getValue();
                GainInfo gainInfo = this.impurity.computeImpurity(statsArray, config);
                if(gainInfo != null) {
                    gainList.add(gainInfo);
                }
            }

            GainInfo maxGainInfo = GainInfo.getGainInfoByMaxGain(gainList);
            if(maxGainInfo == null) {
                // null gain info, set to leaf and continue next stats
                doneNode.setLeaf(true);
                continue;
            }
            populateGainInfoToNode(doneNode, maxGainInfo);

            boolean isLeaf = maxGainInfo.getGain() <= 0d || Node.indexToLevel(doneNode.getId()) == this.maxDepth;
            doneNode.setLeaf(isLeaf);
            if(!doneNode.isLeaf()) {
                boolean leftChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                        || Double.compare(maxGainInfo.getLeftImpurity(), 0d) == 0;
                // such node is just set into isLeaf to true, a new node is created with leaf flag but will be changed
                // to final leaf in later iteration
                int leftIndex = Node.leftIndex(doneNode.getId());
                Node left = new Node(leftIndex, maxGainInfo.getLeftPredict(), maxGainInfo.getLeftImpurity(), true);
                doneNode.setLeft(left);
                if(!leftChildIsLeaf) {
                    this.queue.offer(new TreeNode(treeId, left));
                } else {
                    LOG.debug("Left node {} in tree {} is set to leaf and not submitted to workers", leftIndex, treeId);
                }

                boolean rightChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                        || Double.compare(maxGainInfo.getRightImpurity(), 0d) == 0;

                // such node is just set into isLeaf to true
                int rightIndex = Node.rightIndex(doneNode.getId());
                Node right = new Node(rightIndex, maxGainInfo.getRightPredict(), maxGainInfo.getRightImpurity(), true);
                doneNode.setRight(right);
                if(!rightChildIsLeaf) {
                    this.queue.offer(new TreeNode(treeId, right));
                } else {
                    LOG.debug("Right node {} in tree {} is set to leaf and not submitted to workers", rightIndex, treeId);
                }
            } else {
                LOG.info("Done node {} in tree {} is final set to leaf", doneNode.getId(), treeId);
            }
        }

        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>();
        DTMasterParams masterParams = new DTMasterParams(weightedTrainCount, trainError, weightedValidationCount,
                validationError);
        if(queue.isEmpty()) {
            if(this.isGBDT) {
                Node treeNode = this.trees.get(this.trees.size() - 1).getNode();
                if(this.trees.size() == this.treeNum + this.existingTreeSize) {
                    masterParams.setHalt(true);
                    LOG.info("Queue is empty, training is stopped in iteration {}.", context.getCurrentIteration());
                } else if(treeNode.getLeft() == null && treeNode.getRight() == null) {
                    // if very good performance, here can be some issues, say you'd like to get 5 trees, but in the 2nd
                    // tree, you get one perfect tree, no need continue but warn users about such issue: set
                    // BaggingSampleRate not to 1 can solve such issue to avoid overfit
                    masterParams.setHalt(true);
                    LOG.warn(
                            "Tree is learned 100% well, there must be overfit here, please tune BaggingSampleRate, training is stopped in iteration {}.",
                            context.getCurrentIteration());
                } else {
                    TreeNode newRootNode = new TreeNode(this.trees.size(), new Node(Node.ROOT_INDEX));
                    LOG.info("The {} tree is to be built.", this.trees.size());
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
            List<Integer> depthList = new ArrayList<Integer>();
            for(int i = 0; i < this.trees.size(); i++) {
                // -1 means not work on
                depthList.add(-1);
            }

            while(!queue.isEmpty() && currMem <= this.maxStatsMemory) {
                TreeNode node = this.queue.poll();
                int treeId = node.getTreeId();
                int oldDepth = depthList.get(treeId);
                int currDepth = Node.indexToLevel(node.getNode().getId());
                if(currDepth > oldDepth) {
                    depthList.set(treeId, currDepth);
                }

                List<Integer> subsetFeatures = getSubsamplingFeatures(featureSubsetStrategy);
                node.setFeatures(subsetFeatures);
                currMem += getStatsMem(subsetFeatures);
                todoNodes.put(nodeIndexInGroup, node);
                nodeIndexInGroup += 1;
            }
            masterParams.setTreeDepth(depthList);
            masterParams.setTodoNodes(todoNodes);
            masterParams.setSwitchToNextTree(false);
            LOG.info("Todo node size is {}", todoNodes.size());
        }
        masterParams.setTrees(trees);
        // before master result, do checkpoint according to n iteration set by user
        doCheckPoint(context, masterParams);
        LOG.info("weightedTrainCount {}, weightedValidationCount {}, trainError {}, validationError {}",
                weightedTrainCount, weightedValidationCount, trainError, validationError);
        return masterParams;
    }

    private DTMasterParams rebuildRecoverMasterResultDepthList() {
        DTMasterParams tmpMasterParams = this.cpMasterParams;
        List<Integer> depthList = new ArrayList<Integer>();
        if(isRF || (isGBDT && !isContinuousEnabled)) {
            for(int i = 0; i < this.treeNum; i++) {
                depthList.add(-1);
            }
        } else if(isGBDT && isContinuousEnabled) {
            for(int i = 0; i < this.trees.size(); i++) {
                depthList.add(-1);
            }
        }
        for(Entry<Integer, TreeNode> entry: tmpMasterParams.getTodoNodes().entrySet()) {
            int treeId = entry.getValue().getTreeId();
            int oldDepth = depthList.get(treeId);
            int currDepth = Node.indexToLevel(entry.getValue().getNode().getId());
            if(currDepth > oldDepth) {
                depthList.set(treeId, currDepth);
            }
        }
        tmpMasterParams.setTreeDepth(depthList);
        return tmpMasterParams;
    }

    /**
     * Do checkpoint for master states, this is for master fail over
     */
    private void doCheckPoint(MasterContext<DTMasterParams, DTWorkerParams> context, final DTMasterParams masterParams) {
        boolean isMasterFailOverEnabled = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty("shifu.dt.master.failover", "true"));
        if(!isMasterFailOverEnabled || context.getCurrentIteration() % this.checkpointInterval != 0) {
            return;
        }
        LOG.info("Do checkpoint at hdfs file {}", this.checkpointOutput);
        Thread cpPersistThread = new Thread(new Runnable() {
            @Override
            public void run() {
                writeStatesToHdfs(DTMaster.this.checkpointOutput, masterParams);
            }
        }, "Master checkpoint thread");
        cpPersistThread.setDaemon(true);
        cpPersistThread.start();
    }

    /**
     * Write {@link #trees}, {@link #queue} and MasterParams to HDFS.
     */
    private void writeStatesToHdfs(Path out, DTMasterParams masterParams) {
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(conf).create(out);

            // trees
            int treeLength = trees.size();
            fos.writeInt(treeLength);
            for(TreeNode treeNode: trees) {
                treeNode.write(fos);
            }

            // todo queue
            fos.writeInt(this.queue.size());
            for(TreeNode treeNode: this.queue) {
                treeNode.write(fos);
            }

            // master result
            masterParams.write(fos);
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
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
        List<Integer> tempFeatures = subsetFeatures;
        if(subsetFeatures.size() == 0) {
            tempFeatures = getAllValidFeatures();
        }
        for(Integer columnNum: tempFeatures) {
            ColumnConfig config = this.columnConfigList.get(columnNum);
            // 2 is overhead to avoid oom
            if(config.isNumerical()) {
                statsMem += config.getBinBoundary().size() * this.impurity.getStatsSize() * 8L * 2;
            } else if(config.isCategorical()) {
                statsMem += (config.getBinCategory().size() + 1) * this.impurity.getStatsSize() * 8L * 2;
            }
        }
        // times worker number to avoid oom in master, as combinable DTWorkerParams, use one third of worker number
        statsMem = statsMem * this.workerNumber / 2;
        return statsMem;
    }

    private List<Integer> getAllValidFeatures() {
        List<Integer> features = new ArrayList<Integer>();
        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    features.add(config.getColumnNum());
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                    features.add(config.getColumnNum());
                }
            }
        }
        return features;
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
        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>(treeNum, 1.0f);
        int nodeIndexInGroup = 0;
        List<Integer> depthList = new ArrayList<Integer>();
        DTMasterParams masterParams = new DTMasterParams(trees, todoNodes);
        if(isRF || (isGBDT && !isContinuousEnabled)) {
            for(int i = 0; i < this.treeNum; i++) {
                depthList.add(-1);
            }
            for(TreeNode treeNode: trees) {
                List<Integer> features = getSubsamplingFeatures(this.featureSubsetStrategy);
                treeNode.setFeatures(features);
                todoNodes.put(nodeIndexInGroup, treeNode);
                int treeId = treeNode.getTreeId();
                int oldDepth = depthList.get(treeId);
                int currDepth = Node.indexToLevel(treeNode.getNode().getId());
                if(currDepth > oldDepth) {
                    depthList.set(treeId, currDepth);
                }
                nodeIndexInGroup += 1;
            }
        } else if(isGBDT && isContinuousEnabled) {
            for(int i = 0; i < this.trees.size(); i++) {
                depthList.add(-1);
            }
            List<Integer> features = getSubsamplingFeatures(this.featureSubsetStrategy);
            TreeNode treeNode = trees.get(trees.size() - 1);
            treeNode.setFeatures(features);
            todoNodes.put(nodeIndexInGroup, treeNode);
            int treeId = treeNode.getTreeId();
            int oldDepth = depthList.get(treeId);
            int currDepth = Node.indexToLevel(treeNode.getNode().getId());
            if(currDepth > oldDepth) {
                depthList.set(treeId, currDepth);
            }
            nodeIndexInGroup += 1;
            masterParams.setContinuousRunningStart(true);
            masterParams.setSwitchToNextTree(true);
        }
        masterParams.setTreeDepth(depthList);
        return masterParams;
    }

    private List<Integer> getSubsamplingFeatures(FeatureSubsetStrategy featureSubsetStrategy) {
        switch(featureSubsetStrategy) {
            case HALF:
                return sampleFeaturesForNodeStats(this.allFeatures, this.allFeatures.size() / 2);
            case ONETHIRD:
                return sampleFeaturesForNodeStats(this.allFeatures, this.allFeatures.size() / 3);
            case TWOTHIRDS:
                return sampleFeaturesForNodeStats(this.allFeatures, this.allFeatures.size() * 2 / 3);
            case SQRT:
                return sampleFeaturesForNodeStats(this.allFeatures,
                        (int) (this.allFeatures.size() * Math.sqrt(this.inputNum) / this.inputNum));
            case LOG2:
                return sampleFeaturesForNodeStats(this.allFeatures,
                        (int) (this.allFeatures.size() * Math.log(this.inputNum) / Math.log(2) / this.inputNum));
            case AUTO:
                if(this.treeNum > 1) {
                    return sampleFeaturesForNodeStats(this.allFeatures, this.allFeatures.size() / 2);
                } else {
                    return new ArrayList<Integer>();
                }
            case ALL:
            default:
                return new ArrayList<Integer>();
        }
    }

    private List<Integer> getAllFeatureList(List<ColumnConfig> columnConfigList, boolean isAfterVarSelect) {
        List<Integer> features = new ArrayList<Integer>();
        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    features.add(config.getColumnNum());
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                    features.add(config.getColumnNum());
                }
            }
        }
        return features;
    }

    private List<Integer> sampleFeaturesForNodeStats(List<Integer> allFeatures, int sample) {
        List<Integer> features = new ArrayList<Integer>(sample);
        for(int i = 0; i < sample; i++) {
            features.add(allFeatures.get(i));
        }

        Random random = new Random();
        for(int i = sample; i < allFeatures.size(); i++) {
            int replacementIndex = (int) (random.nextDouble() * i);
            if(replacementIndex >= 0 && replacementIndex < sample) {
                features.set(replacementIndex, allFeatures.get(i));
            }
        }
        return features;
    }

    @Override
    public void init(MasterContext<DTMasterParams, DTWorkerParams> context) {
        Properties props = context.getProps();

        // init model config and column config list at first
        SourceType sourceType;
        try {
            sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // worker number is used to estimate nodes per iteration for stats
        this.workerNumber = NumberFormatUtils.getInt(props.getProperty(GuaguaConstants.GUAGUA_WORKER_NUMBER), true);

        // check if variables are set final selected
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.inputNum = inputOutputIndex[0] + inputOutputIndex[1];
        this.isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        this.allFeatures = this.getAllFeatureList(columnConfigList, isAfterVarSelect);

        int trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams());
        Map<String, Object> validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }
        // tree related parameters initialization
        this.featureSubsetStrategy = FeatureSubsetStrategy.of(validParams.get("FeatureSubsetStrategy").toString());
        this.maxDepth = Integer.valueOf(validParams.get("MaxDepth").toString());
        assert this.maxDepth > 0 && this.maxDepth <= 20;
        this.maxStatsMemory = Long.valueOf(validParams.get("MaxStatsMemoryMB").toString()) * 1024 * 1024;
        // assert this.maxStatsMemory <= Math.min(Runtime.getRuntime().maxMemory() * 0.6, 800 * 1024 * 1024L);
        this.treeNum = Integer.valueOf(validParams.get("TreeNum").toString());;
        this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        this.isGBDT = ALGORITHM.GBT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        if(this.isGBDT) {
            // learning rate only effective in gbdt
            this.learningRate = Double.valueOf(validParams.get(NNTrainer.LEARNING_RATE).toString());
        }
        String imStr = validParams.get("Impurity").toString();
        int numClasses = 2;
        if(this.modelConfig.isClassification()) {
            numClasses = this.modelConfig.getTags().size();
        }

        int minInstancesPerNode = Integer.valueOf(validParams.get("MinInstancesPerNode").toString());
        double minInfoGain = Double.valueOf(validParams.get("MinInfoGain").toString());
        if(imStr.equalsIgnoreCase("entropy")) {
            impurity = new Entropy(numClasses, minInstancesPerNode, minInfoGain);
        } else if(imStr.equalsIgnoreCase("gini")) {
            impurity = new Gini(numClasses, minInstancesPerNode, minInfoGain);
        } else {
            impurity = new Variance(minInstancesPerNode, minInfoGain);
        }

        this.checkpointInterval = NumberFormatUtils.getInt(context.getProps().getProperty(
                CommonConstants.SHIFU_DT_MASTER_CHECKPOINT_INTERVAL, "20"));
        this.checkpointOutput = new Path(context.getProps().getProperty(
                CommonConstants.SHIFU_DT_MASTER_CHECKPOINT_FOLDER, "tmp/cp_" + context.getAppId()));

        this.conf = new Configuration();

        this.isContinuousEnabled = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));

        LOG.info("Master init params: isAfterVarSel={}, featureSubsetStrategy={}, maxDepth={}, maxStatsMemory={}, "
                + "treeNum={}, impurity={}, workerNumber={}, minInstancesPerNode={}, minInfoGain={}, isRF={}, "
                + "isGBDT={}, isContinuousEnabled={}", isAfterVarSelect, featureSubsetStrategy, maxDepth,
                maxStatsMemory, treeNum, imStr, this.workerNumber, minInstancesPerNode, minInfoGain, this.isRF,
                this.isGBDT, this.isContinuousEnabled);
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
                if(isContinuousEnabled) {
                    TreeModel existingModel;
                    try {
                        Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
                        existingModel = (TreeModel) CommonUtils.loadModel(modelConfig, columnConfigList, modelPath,
                                ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource()));
                        if(existingModel == null) {
                            this.trees = new ArrayList<TreeNode>(treeNum);
                            this.trees.add(new TreeNode(0, new Node(Node.ROOT_INDEX)));
                            LOG.info("Starting to train model from scratch.");
                        } else {
                            this.trees = existingModel.getTrees();
                            this.existingTreeSize = this.trees.size();
                            this.trees.add(new TreeNode(this.existingTreeSize, new Node(Node.ROOT_INDEX)));
                            LOG.info("Starting to train model from existing model {} with existing trees {}.",
                                    modelPath, existingTreeSize);
                        }
                    } catch (IOException e) {
                        throw new GuaguaRuntimeException(e);
                    }
                } else {
                    this.trees = new ArrayList<TreeNode>(treeNum);
                    // for GBDT, initialize the first tree. trees are trained sequentially
                    this.trees.add(new TreeNode(0, new Node(Node.ROOT_INDEX)));
                }
            }
        } else {
            // recover all states once master is fail over
            LOG.info("Recover master stats from cp file {}", this.checkpointOutput);
            recoverMasterStats(sourceType);
        }
    }

    private void recoverMasterStats(SourceType sourceType) {
        FSDataInputStream stream = null;
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        try {
            stream = fs.open(this.checkpointOutput);
            int treeSize = stream.readInt();
            this.trees = new ArrayList<TreeNode>(treeSize);
            for(int i = 0; i < treeSize; i++) {
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(stream);
                this.trees.add(treeNode);
            }

            int queueSize = stream.readInt();
            for(int i = 0; i < queueSize; i++) {
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(stream);
                this.queue.offer(treeNode);
            }

            this.cpMasterParams = new DTMasterParams();
            this.cpMasterParams.readFields(stream);
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        } finally {
            if(stream != null) {
                org.apache.commons.io.IOUtils.closeQuietly(stream);
            }
        }
    }
}
