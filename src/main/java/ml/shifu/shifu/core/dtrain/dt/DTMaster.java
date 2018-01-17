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
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;

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
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.FeatureSubsetStrategy;
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
 * In each iteration, update node statistics and determine best split which is used for tree node split. Besides node
 * statistics, error and count info are also collected for metrics display.
 * 
 * <p>
 * Each iteration, new node group with nodes in limited estimated memory consumption are sent out to all workers for
 * feature statistics.
 * 
 * <p>
 * For gradient boost decision tree, each time a tree is updated and after one tree is finalized, then start a new tree.
 * Both random forest and gradient boost decision trees are all stored in {@link #trees}.
 * 
 * <p>
 * Terminal condition: for random forest, just to collect all nodes in all trees from all workers. Terminal condition is
 * all trees cannot be split. If one tree cannot be split with threshold count and meaningful impurity, one tree if
 * finalized and stopped update. For gradient boost decision tree, each time only one tree is trained, if last tree
 * cannot be split, training is stopped. Early stop feature is enabled by validationTolerance in train part.
 * 
 * <p>
 * In current {@link DTMaster}, there are states like {@link #trees} and {@link #toDoQueue}. All stats can be recovered
 * once master is done. Such states are being check-pointed to HDFS for fault tolerance.
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
     * Feature sub sampling strategy, this is combined with {@link #featureSubsetRate}, if
     * {@link #featureSubsetStrategy} is null, use {@link #featureSubsetRate}. Otherwise use
     * {@link #featureSubsetStrategy}.
     */
    private FeatureSubsetStrategy featureSubsetStrategy = FeatureSubsetStrategy.ALL;

    /**
     * FeatureSubsetStrategy in train#params can be set to double or text, if double, use current double value but
     * {@link #featureSubsetStrategy} is set to null.
     */
    private double featureSubsetRate;

    /**
     * Max depth of a tree, by default is 10.
     */
    private int maxDepth;

    /**
     * Max leaves of a tree, by default is -1. If maxLeaves is set > 0, level-wise tree building is enabled no matter
     * {@link #maxDepth} set to what value.
     */
    private int maxLeaves = -1;

    /**
     * maxLeaves >= -1, then isLeafWise set to true, else level-wise tree building.
     */
    private boolean isLeafWise = false;

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
    private double learningRate;

    /**
     * How many workers, this is used for memory usage
     */
    private int workerNumber;

    /**
     * Input features numbers
     */
    private int inputNum;

    /**
     * Cache all features with feature index for searching
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
     * Every checkpoint interval, do checkpoint to save {@link #trees} and {@link #toDoQueue} and MasterParams in that
     * iteration.
     */
    @SuppressWarnings("unused")
    private int checkpointInterval;

    /**
     * Checkpoint output HDFS file
     */
    private Path checkpointOutput;

    /**
     * Common conf to avoid new Configuration
     */
    @SuppressWarnings("unused")
    private Configuration conf;

    /**
     * Checkpoint master params, if only recover queue in fail over, some todo nodes in master result will be ignored.
     * This is used to recover whole states of that iteration. In {@link #doCompute(MasterContext)}, check if
     * {@link #cpMasterParams} is null, if not, directly return this one and send {@link #cpMasterParams} to null to
     * avoid next iteration to send it again.
     */
    private DTMasterParams cpMasterParams;

    /**
     * Max batch split size in leaf-wise tree growth.; This only works well when {@link #isLeafWise} = true.
     */
    private int maxBatchSplitSize = 16;

    /**
     * DTEarlyStopDecider will decide automatic whether it need further training, this only for GBDT.
     */
    private DTEarlyStopDecider dtEarlyStopDecider;

    /**
     * If early stop is enabled or not, by default false.
     */
    private boolean enableEarlyStop = false;

    /**
     * Validation tolerance which is for early stop, by default it is 0d which means early stop is not enabled.
     */
    private double validationTolerance = 0d;

    /**
     * Random generator for get sampling features per each iteration.
     */
    private Random featureSamplingRandom = new Random();

    /**
     * The best validation error for error computing
     */
    private double bestValidationError = Double.MAX_VALUE;

    // ############################################################################################################
    // ## There parts are states, for fail over such instances should be recovered in {@link #init(MasterContext)}
    // ############################################################################################################

    /**
     * All trees trained in this master
     */
    private List<TreeNode> trees;

    /**
     * TreeNode with splits will be add to this queue and after that, split a batch of nodes at the same iteration; this
     * only works well when {@link #isLeafWise} = true.
     */
    private Queue<TreeNode> toSplitQueue;

    /**
     * TreeNodes needed to be collected statistics from workers.
     */
    private Queue<TreeNode> toDoQueue;

    @Override
    public DTMasterParams doCompute(MasterContext<DTMasterParams, DTWorkerParams> context) {
        if(context.isFirstIteration()) {
            return buildInitialMasterParams();
        }

        if(this.cpMasterParams != null) {
            DTMasterParams tmpMasterParams = rebuildRecoverMasterResultDepthList();
            // set it to null to avoid send it in next iteration
            this.cpMasterParams = null;
            if(this.isGBDT) {
                // don't need to send full trees because worker will get existing models from HDFS
                // only set last tree to do node stats, no need check switch to next tree because of message may be send
                // to worker already
                tmpMasterParams.setTrees(trees.subList(trees.size() - 1, trees.size()));
                // set tmp trees for DTOutput
                tmpMasterParams.setTmpTrees(this.trees);
            }
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
                // set to null after merging, release memory at the earliest stage
                params.setNodeStatsMap(null);
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
            populateGainInfoToNode(treeId, doneNode, maxGainInfo);

            if(this.isLeafWise) {
                boolean isNotSplit = maxGainInfo.getGain() <= 0d;
                if(!isNotSplit) {
                    this.toSplitQueue.offer(new TreeNode(treeId, doneNode));
                } else {
                    LOG.info("Node {} in tree {} is not to be split", doneNode.getId(), treeId);
                }
            } else {
                boolean isLeaf = maxGainInfo.getGain() <= 0d || Node.indexToLevel(doneNode.getId()) == this.maxDepth;
                doneNode.setLeaf(isLeaf);
                // level-wise is to split node when stats is ready
                splitNodeForLevelWisedTree(isLeaf, treeId, doneNode);
            }
        }

        if(this.isLeafWise) {
            // get node in toSplitQueue and split
            int currSplitIndex = 0;
            while(!toSplitQueue.isEmpty() && currSplitIndex < this.maxBatchSplitSize) {
                TreeNode treeNode = this.toSplitQueue.poll();
                splitNodeForLeafWisedTree(treeNode.getTreeId(), treeNode.getNode());
            }
        }

        Map<Integer, TreeNode> todoNodes = new HashMap<Integer, TreeNode>();
        double averageValidationError = validationError / weightedValidationCount;
        if(this.isGBDT && this.dtEarlyStopDecider != null && averageValidationError > 0) {
            this.dtEarlyStopDecider.add(averageValidationError);
            averageValidationError = this.dtEarlyStopDecider.getCurrentAverageValue();
        }

        boolean vtTriggered = false;
        // if validationTolerance == 0d, means vt check is not enabled
        if(validationTolerance > 0d
                && Math.abs(this.bestValidationError - averageValidationError) < this.validationTolerance
                        * averageValidationError) {
            LOG.debug("Debug: bestValidationError {}, averageValidationError {}, validationTolerance {}",
                    bestValidationError, averageValidationError, validationTolerance);
            vtTriggered = true;
        }

        if(averageValidationError < this.bestValidationError) {
            this.bestValidationError = averageValidationError;
        }

        // validation error is averageValidationError * weightedValidationCount because of here averageValidationError
        // is divided by validation count.
        DTMasterParams masterParams = new DTMasterParams(weightedTrainCount, trainError, weightedValidationCount,
                averageValidationError * weightedValidationCount);

        if(toDoQueue.isEmpty()) {
            if(this.isGBDT) {
                TreeNode treeNode = this.trees.get(this.trees.size() - 1);
                Node node = treeNode.getNode();
                if(this.trees.size() >= this.treeNum) {
                    // if all trees including trees read from existing model over treeNum, stop the whole process.
                    masterParams.setHalt(true);
                    LOG.info("Queue is empty, training is stopped in iteration {}.", context.getCurrentIteration());
                } else if(node.getLeft() == null && node.getRight() == null) {
                    // if very good performance, here can be some issues, say you'd like to get 5 trees, but in the 2nd
                    // tree, you get one perfect tree, no need continue but warn users about such issue: set
                    // BaggingSampleRate not to 1 can solve such issue to avoid overfit
                    masterParams.setHalt(true);
                    LOG.warn(
                            "Tree is learned 100% well, there must be overfit here, please tune BaggingSampleRate, training is stopped in iteration {}.",
                            context.getCurrentIteration());
                } else if(this.dtEarlyStopDecider != null
                        && (this.enableEarlyStop && this.dtEarlyStopDecider.canStop())) {
                    masterParams.setHalt(true);
                    LOG.info("Early stop identified, training is stopped in iteration {}.",
                            context.getCurrentIteration());
                } else if(vtTriggered) {
                    LOG.info("Early stop training by validation tolerance.");
                    masterParams.setHalt(true);
                } else {
                    // set first tree to true even after ROOT node is set in next tree
                    masterParams.setFirstTree(this.trees.size() == 1);
                    // finish current tree, no need features information
                    treeNode.setFeatures(null);
                    TreeNode newRootNode = new TreeNode(this.trees.size(), new Node(Node.ROOT_INDEX), this.learningRate);
                    LOG.info("The {} tree is to be built.", this.trees.size());
                    this.trees.add(newRootNode);
                    newRootNode.setFeatures(getSubsamplingFeatures(this.featureSubsetStrategy, this.featureSubsetRate));
                    // only one node
                    todoNodes.put(0, newRootNode);
                    masterParams.setTodoNodes(todoNodes);
                    // set switch flag
                    masterParams.setSwitchToNextTree(true);
                }
            } else {
                // for rf
                masterParams.setHalt(true);
                LOG.info("Queue is empty, training is stopped in iteration {}.", context.getCurrentIteration());
            }
        } else {
            int nodeIndexInGroup = 0;
            long currMem = 0L;
            List<Integer> depthList = new ArrayList<Integer>();
            if(this.isGBDT) {
                depthList.add(-1);
            }

            if(isRF) {
                for(int i = 0; i < this.trees.size(); i++) {
                    depthList.add(-1);
                }
            }

            while(!toDoQueue.isEmpty() && currMem <= this.maxStatsMemory) {
                TreeNode node = this.toDoQueue.poll();
                int treeId = node.getTreeId();
                int oldDepth = this.isGBDT ? depthList.get(0) : depthList.get(treeId);
                int currDepth = Node.indexToLevel(node.getNode().getId());
                if(currDepth > oldDepth) {
                    if(this.isGBDT) {
                        // gbdt only for last depth
                        depthList.set(0, currDepth);
                    } else {
                        depthList.set(treeId, currDepth);
                    }
                }

                List<Integer> subsetFeatures = getSubsamplingFeatures(this.featureSubsetStrategy,
                        this.featureSubsetRate);
                node.setFeatures(subsetFeatures);
                currMem += getStatsMem(subsetFeatures);
                todoNodes.put(nodeIndexInGroup, node);
                nodeIndexInGroup += 1;
            }
            masterParams.setTreeDepth(depthList);
            masterParams.setTodoNodes(todoNodes);
            masterParams.setSwitchToNextTree(false);
            masterParams.setContinuousRunningStart(false);
            masterParams.setFirstTree(this.trees.size() == 1);
            LOG.info("Todo node size is {}", todoNodes.size());
        }

        if(this.isGBDT) {
            if(masterParams.isSwitchToNextTree()) {
                // send last full growth tree and current todo ROOT node tree
                masterParams.setTrees(trees.subList(trees.size() - 2, trees.size()));
            } else {
                // only send current trees
                masterParams.setTrees(trees.subList(trees.size() - 1, trees.size()));
            }
        }

        if(this.isRF) {
            // for rf, reset trees sent to workers for only trees with todo nodes, this saves message space. While
            // elements in todoTrees are also the same reference in this.trees, reuse the same object to save memory.
            if(masterParams.getTreeDepth().size() == this.trees.size()) {
                // if normal iteration
                List<TreeNode> todoTrees = new ArrayList<TreeNode>();
                for(int i = 0; i < trees.size(); i++) {
                    if(masterParams.getTreeDepth().get(i) >= 0) {
                        // such tree in current iteration treeDepth is not -1, add it to todoTrees.
                        todoTrees.add(trees.get(i));
                    } else {
                        // mock a TreeNode instance to make sure no surprise in further serialization. In fact
                        // meaningless.
                        todoTrees.add(new TreeNode(i, new Node(Node.INVALID_INDEX), 1d));
                    }
                }
                masterParams.setTrees(todoTrees);
            } else {
                // if last iteration without maxDepthList
                masterParams.setTrees(trees);
            }
        }
        if(this.isGBDT) {
            // set tmp trees to DTOutput
            masterParams.setTmpTrees(this.trees);
        }

        if(context.getCurrentIteration() % 100 == 0) {
            // every 100 iterations do gc explicitly to avoid one case:
            // mapper memory is 2048M and final in our cluster, if -Xmx is 2G, then occasionally oom issue.
            // to fix this issue: 1. set -Xmx to 1800m; 2. call gc to drop unused memory at early stage.
            // this is ugly and if it is stable with 1800m, this line should be removed
            Thread gcThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    System.gc();
                }
            });
            gcThread.setDaemon(true);
            gcThread.start();
        }

        // before master result, do checkpoint according to n iteration set by user
        doCheckPoint(context, masterParams, context.getCurrentIteration());

        LOG.debug("weightedTrainCount {}, weightedValidationCount {}, trainError {}, validationError {}",
                weightedTrainCount, weightedValidationCount, trainError, validationError);
        return masterParams;
    }

    /**
     * Split node into left and right for leaf-wised tree growth, doneNode should be populated by
     * {@link #populateGainInfoToNode(Node, GainInfo)}.
     */
    private void splitNodeForLeafWisedTree(int treeId, Node doneNode) {
        boolean isOverMaxLeaves = this.trees.get(treeId).getNodeNum() + 1 > this.maxLeaves;
        boolean canSplit = !isOverMaxLeaves && Double.compare(doneNode.getLeftImpurity(), 0d) != 0;
        // if can split left, at the same time create left and right node
        if(canSplit) {
            int leftIndex = Node.leftIndex(doneNode.getId());
            Node left = new Node(leftIndex, doneNode.getLeftPredict(), doneNode.getLeftImpurity(), true);

            doneNode.setLeft(left);
            this.trees.get(treeId).incrNodeNum();
            this.toDoQueue.offer(new TreeNode(treeId, left));

            int rightIndex = Node.rightIndex(doneNode.getId());
            Node right = new Node(rightIndex, doneNode.getRightPredict(), doneNode.getRightImpurity(), true);

            doneNode.setRight(right);
            this.trees.get(treeId).incrNodeNum();
            this.toDoQueue.offer(new TreeNode(treeId, right));
        }
    }

    /**
     * Split node into left and right for level-wised tree growth, doneNode should be populated by
     * {@link #populateGainInfoToNode(Node, GainInfo)}
     */
    private void splitNodeForLevelWisedTree(boolean isLeaf, int treeId, Node doneNode) {
        if(!isLeaf) {
            boolean leftChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                    || Double.compare(doneNode.getLeftImpurity(), 0d) == 0;
            // such node is just set into isLeaf to true, a new node is created with leaf flag but will be
            // changed to final leaf in later iteration
            int leftIndex = Node.leftIndex(doneNode.getId());
            Node left = new Node(leftIndex, doneNode.getLeftPredict(), doneNode.getLeftImpurity(), true);
            doneNode.setLeft(left);
            // update nodeNum
            this.trees.get(treeId).incrNodeNum();

            if(!leftChildIsLeaf) {
                this.toDoQueue.offer(new TreeNode(treeId, left));
            } else {
                LOG.debug("Left node {} in tree {} is set to leaf and not submitted to workers", leftIndex, treeId);
            }

            boolean rightChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                    || Double.compare(doneNode.getRightImpurity(), 0d) == 0;

            // such node is just set into isLeaf to true
            int rightIndex = Node.rightIndex(doneNode.getId());
            Node right = new Node(rightIndex, doneNode.getRightPredict(), doneNode.getRightImpurity(), true);

            doneNode.setRight(right);
            // update nodeNum
            this.trees.get(treeId).incrNodeNum();

            if(!rightChildIsLeaf) {
                this.toDoQueue.offer(new TreeNode(treeId, right));
            } else {
                LOG.debug("Right node {} in tree {} is set to leaf and not submitted to workers", rightIndex, treeId);
            }
        } else {
            LOG.info("Done node {} in tree {} is final set to leaf", doneNode.getId(), treeId);
        }
    }

    private DTMasterParams rebuildRecoverMasterResultDepthList() {
        DTMasterParams tmpMasterParams = this.cpMasterParams;
        List<Integer> depthList = new ArrayList<Integer>();
        if(isRF) {
            for(int i = 0; i < this.treeNum; i++) {
                depthList.add(-1);
            }
        } else if(isGBDT) {
            depthList.add(-1);
        }
        for(Entry<Integer, TreeNode> entry: tmpMasterParams.getTodoNodes().entrySet()) {
            int treeId = entry.getValue().getTreeId();
            int oldDepth = isGBDT ? depthList.get(0) : depthList.get(treeId);
            int currDepth = Node.indexToLevel(entry.getValue().getNode().getId());
            if(currDepth > oldDepth) {
                if(isGBDT) {
                    depthList.set(0, currDepth);
                }
                if(isRF) {
                    depthList.set(treeId, currDepth);
                }
            }
        }
        tmpMasterParams.setTreeDepth(depthList);
        return tmpMasterParams;
    }

    /**
     * Do checkpoint for master states, this is for master fail over
     */
    private void doCheckPoint(final MasterContext<DTMasterParams, DTWorkerParams> context,
            final DTMasterParams masterParams, int iteration) {
        String intervalStr = context.getProps().getProperty(CommonConstants.SHIFU_TREE_CHECKPOINT_INTERVAL);
        int interval = 100;
        try {
            interval = Integer.parseInt(intervalStr);
        } catch (Exception ignore) {
        }

        // only do checkpoint in interval configured.
        if(iteration != 0 && iteration % interval != 0) {
            return;
        }

        LOG.info("Do checkpoint at hdfs file {} at iteration {}.", this.checkpointOutput, iteration);
        final Queue<TreeNode> finalTodoQueue = this.toDoQueue;
        final Queue<TreeNode> finalToSplitQueue = this.toSplitQueue;
        final boolean finalIsLeaf = this.isLeafWise;
        final List<TreeNode> finalTrees = this.trees;

        Thread cpPersistThread = new Thread(new Runnable() {
            @Override
            public void run() {
                long start = System.currentTimeMillis();
                writeStatesToHdfs(DTMaster.this.checkpointOutput, masterParams, finalTrees, finalIsLeaf,
                        finalTodoQueue, finalToSplitQueue);
                LOG.info("Do checkpoint in iteration {} with run time {}", context.getCurrentIteration(),
                        (System.currentTimeMillis() - start));
            }
        }, "Master checkpoint thread");
        cpPersistThread.setDaemon(true);
        cpPersistThread.start();
    }

    /**
     * Write {@link #trees}, {@link #toDoQueue} and MasterParams to HDFS.
     */
    private void writeStatesToHdfs(Path out, DTMasterParams masterParams, List<TreeNode> trees, boolean isLeafWise,
            Queue<TreeNode> toDoQueue, Queue<TreeNode> toSplitQueue) {
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);

            // trees
            int treeLength = trees.size();
            fos.writeInt(treeLength);
            for(TreeNode treeNode: trees) {
                treeNode.write(fos);
            }

            // todo queue
            fos.writeInt(toDoQueue.size());
            for(TreeNode treeNode: toDoQueue) {
                treeNode.write(fos);
            }

            if(isLeafWise && toSplitQueue != null) {
                fos.writeInt(toSplitQueue.size());
                for(TreeNode treeNode: toSplitQueue) {
                    treeNode.write(fos);
                }
            }

            // master result
            masterParams.write(fos);
        } catch (Throwable e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
            fos = null;
        }
    }

    private void populateGainInfoToNode(int treeId, Node doneNode, GainInfo maxGainInfo) {
        doneNode.setPredict(maxGainInfo.getPredict());
        doneNode.setSplit(maxGainInfo.getSplit());
        doneNode.setGain(maxGainInfo.getGain());
        doneNode.setImpurity(maxGainInfo.getImpurity());
        doneNode.setLeftImpurity(maxGainInfo.getLeftImpurity());
        doneNode.setRightImpurity(maxGainInfo.getRightImpurity());
        doneNode.setLeftPredict(maxGainInfo.getLeftPredict());
        doneNode.setRightPredict(maxGainInfo.getRightPredict());
        doneNode.setWgtCnt(maxGainInfo.getWgtCnt());

        if(Node.isRootNode(doneNode)) {
            this.trees.get(treeId).setRootWgtCnt(maxGainInfo.getWgtCnt());
        } else {
            double rootWgtCnt = this.trees.get(treeId).getRootWgtCnt();
            doneNode.setWgtCntRatio(maxGainInfo.getWgtCnt() / rootWgtCnt);
        }
    }

    private long getStatsMem(List<Integer> subsetFeatures) {
        long statsMem = 0L;
        List<Integer> tempFeatures = subsetFeatures;
        if(subsetFeatures.size() == 0) {
            tempFeatures = getAllFeatureList(this.columnConfigList, this.isAfterVarSelect);
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
        if(isRF) {
            // for RF, all trees should be set depth
            for(int i = 0; i < this.treeNum; i++) {
                depthList.add(-1);
            }
            for(TreeNode treeNode: trees) {
                List<Integer> features = getSubsamplingFeatures(this.featureSubsetStrategy, this.featureSubsetRate);
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
            // For RF, each time send whole trees
            masterParams.setTrees(this.trees);
        } else if(isGBDT) {
            // for gbdt, only store depth of last tree
            depthList.add(-1);
            List<Integer> features = getSubsamplingFeatures(this.featureSubsetStrategy, this.featureSubsetRate);
            TreeNode treeNode = trees.get(trees.size() - 1); // only for last tree
            treeNode.setFeatures(features);
            todoNodes.put(nodeIndexInGroup, treeNode);
            int oldDepth = depthList.get(0);
            int currDepth = Node.indexToLevel(treeNode.getNode().getId());
            if(currDepth > oldDepth) {
                depthList.set(0, currDepth);
            }
            nodeIndexInGroup += 1;
            // isContinuousEnabled true means this is the first iteration for continuous model training, worker should
            // recover predict value from existing models
            masterParams.setContinuousRunningStart(this.isContinuousEnabled);
            // switch to next new tree for only ROOT node stats
            masterParams.setSwitchToNextTree(true);
            // if current tree is the first tree
            masterParams.setFirstTree(this.trees.size() == 1);
            // gbdt only send last tree to workers
            if(this.trees.size() > 0) {
                masterParams.setTrees(this.trees.subList(this.trees.size() - 1, this.trees.size()));
            }
            // tmp trees will not send to workers, just to DTOutput for model saving
            masterParams.setTmpTrees(this.trees);
        }
        masterParams.setTreeDepth(depthList);
        return masterParams;
    }

    private List<Integer> getSubsamplingFeatures(FeatureSubsetStrategy featureSubsetStrategy, double featureSubsetRate) {
        if(featureSubsetStrategy == null) {
            return sampleFeaturesForNodeStats(this.allFeatures, (int) (this.allFeatures.size() * featureSubsetRate));
        } else {
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
    }

    private List<Integer> getAllFeatureList(List<ColumnConfig> columnConfigList, boolean isAfterVarSelect) {
        List<Integer> features = new ArrayList<Integer>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    // only select numerical feature with getBinBoundary().size() larger than 1
                    // or categorical feature with getBinCategory().size() larger than 0
                    if((config.isNumerical() && config.getBinBoundary().size() > 1)
                            || (config.isCategorical() && config.getBinCategory().size() > 0)) {
                        features.add(config.getColumnNum());
                    }
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config, hasCandidates)) {
                    // only select numerical feature with getBinBoundary().size() larger than 1
                    // or categorical feature with getBinCategory().size() larger than 0
                    if((config.isNumerical() && config.getBinBoundary().size() > 1)
                            || (config.isCategorical() && config.getBinCategory().size() > 0)) {
                        features.add(config.getColumnNum());
                    }
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

        for(int i = sample; i < allFeatures.size(); i++) {
            int replacementIndex = (int) (featureSamplingRandom.nextDouble() * i);
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
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        // cache all feature list for sampling features
        this.allFeatures = this.getAllFeatureList(columnConfigList, isAfterVarSelect);

        int trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        // If grid search, select valid paramters, if not parameters is what in ModelConfig.json
        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain()
                .getGridConfigFileContent());
        Map<String, Object> validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }

        Object vtObj = validParams.get("ValidationTolerance");
        if(vtObj != null) {
            try {
                validationTolerance = Double.parseDouble(vtObj.toString());
                LOG.warn("Validation by tolerance is enabled with value {}.", validationTolerance);
            } catch (NumberFormatException ee) {
                validationTolerance = 0d;
                LOG.warn(
                        "Validation by tolerance isn't enabled because of non numerical value of ValidationTolerance: {}.",
                        vtObj);
            }
        } else {
            LOG.warn("Validation by tolerance isn't enabled.");
        }

        // tree related parameters initialization
        Object fssObj = validParams.get("FeatureSubsetStrategy");
        if(fssObj != null) {
            try {
                this.featureSubsetRate = Double.parseDouble(fssObj.toString());
                // no need validate featureSubsetRate is in (0,1], as already validated in ModelInspector
                this.featureSubsetStrategy = null;
            } catch (NumberFormatException ee) {
                this.featureSubsetStrategy = FeatureSubsetStrategy.of(fssObj.toString());
            }
        } else {
            LOG.warn("FeatureSubsetStrategy is not set, set to TWOTHRIDS by default in DTMaster.");
            this.featureSubsetStrategy = FeatureSubsetStrategy.TWOTHIRDS;
            this.featureSubsetRate = 0;
        }

        // max depth
        Object maxDepthObj = validParams.get("MaxDepth");
        if(maxDepthObj != null) {
            this.maxDepth = Integer.valueOf(maxDepthObj.toString());
        } else {
            this.maxDepth = 10;
        }

        // max leaves which is used for leaf-wised tree building, TODO add more benchmarks
        Object maxLeavesObj = validParams.get("MaxLeaves");
        if(maxLeavesObj != null) {
            this.maxLeaves = Integer.valueOf(maxLeavesObj.toString());
        } else {
            this.maxLeaves = -1;
        }

        // enable leaf wise tree building once maxLeaves is configured
        if(this.maxLeaves > 0) {
            this.isLeafWise = true;
        }

        // maxBatchSplitSize means each time split # of batch nodes
        Object maxBatchSplitSizeObj = validParams.get("MaxBatchSplitSize");
        if(maxBatchSplitSizeObj != null) {
            this.maxBatchSplitSize = Integer.valueOf(maxBatchSplitSizeObj.toString());
        } else {
            // by default split 32 at most in a batch
            this.maxBatchSplitSize = 32;
        }

        assert this.maxDepth > 0 && this.maxDepth <= 20;

        // hide in parameters, this to avoid OOM issue for each iteration
        Object maxStatsMemoryMB = validParams.get("MaxStatsMemoryMB");
        if(maxStatsMemoryMB != null) {
            this.maxStatsMemory = Long.valueOf(validParams.get("MaxStatsMemoryMB").toString()) * 1024 * 1024;
            if(this.maxStatsMemory > ((2L * Runtime.getRuntime().maxMemory()) / 3)) {
                // if >= 2/3 max memory, take 2/3 max memory to avoid OOM
                this.maxStatsMemory = ((2L * Runtime.getRuntime().maxMemory()) / 3);
            }
        } else {
            // by default it is 1/2 of heap, about 1.5G setting in current Shifu
            this.maxStatsMemory = Runtime.getRuntime().maxMemory() / 2L;
        }

        // assert this.maxStatsMemory <= Math.min(Runtime.getRuntime().maxMemory() * 0.6, 800 * 1024 * 1024L);
        this.treeNum = Integer.valueOf(validParams.get("TreeNum").toString());
        this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        this.isGBDT = ALGORITHM.GBT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        if(this.isGBDT) {
            // learning rate only effective in gbdt
            this.learningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());
        }

        // initialize impurity type according to regression or classfication
        String imStr = validParams.get("Impurity").toString();
        int numClasses = 2;
        if(this.modelConfig.isClassification()) {
            numClasses = this.modelConfig.getTags().size();
        }
        // these two parameters is to stop tree growth parameters
        int minInstancesPerNode = Integer.valueOf(validParams.get("MinInstancesPerNode").toString());
        double minInfoGain = Double.valueOf(validParams.get("MinInfoGain").toString());
        if(imStr.equalsIgnoreCase("entropy")) {
            impurity = new Entropy(numClasses, minInstancesPerNode, minInfoGain);
        } else if(imStr.equalsIgnoreCase("gini")) {
            impurity = new Gini(numClasses, minInstancesPerNode, minInfoGain);
        } else {
            impurity = new Variance(minInstancesPerNode, minInfoGain);
        }

        // checkpoint folder and interval (every # iterations to do checkpoint)
        this.checkpointInterval = NumberFormatUtils.getInt(context.getProps().getProperty(
                CommonConstants.SHIFU_DT_MASTER_CHECKPOINT_INTERVAL, "20"));
        this.checkpointOutput = new Path(context.getProps().getProperty(
                CommonConstants.SHIFU_DT_MASTER_CHECKPOINT_FOLDER, "tmp/cp_" + context.getAppId()));

        // cache conf to avoid new
        this.conf = new Configuration();

        // if continous model training is enabled
        this.isContinuousEnabled = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));

        this.dtEarlyStopDecider = new DTEarlyStopDecider(this.maxDepth);
        if(validParams.containsKey("EnableEarlyStop")
                && Boolean.valueOf(validParams.get("EnableEarlyStop").toString().toLowerCase())) {
            this.enableEarlyStop = true;
        }

        LOG.info(
                "Master init params: isAfterVarSel={}, featureSubsetStrategy={}, featureSubsetRate={} maxDepth={}, maxStatsMemory={}, "
                        + "treeNum={}, impurity={}, workerNumber={}, minInstancesPerNode={}, minInfoGain={}, isRF={}, "
                        + "isGBDT={}, isContinuousEnabled={}, enableEarlyStop={}.", isAfterVarSelect,
                featureSubsetStrategy, this.featureSubsetRate, maxDepth, maxStatsMemory, treeNum, imStr,
                this.workerNumber, minInstancesPerNode, minInfoGain, this.isRF, this.isGBDT, this.isContinuousEnabled,
                this.enableEarlyStop);

        this.toDoQueue = new LinkedList<TreeNode>();

        if(this.isLeafWise) {
            this.toSplitQueue = new PriorityQueue<TreeNode>(64, new Comparator<TreeNode>() {
                @Override
                public int compare(TreeNode o1, TreeNode o2) {
                    return Double.compare(o2.getNode().getWgtCntRatio() * o2.getNode().getGain(), o1.getNode()
                            .getWgtCntRatio() * o1.getNode().getGain());
                }
            });
        }
        // initialize trees
        if(context.isFirstIteration()) {
            if(this.isRF) {
                // for random forest, trees are trained in parallel
                this.trees = new CopyOnWriteArrayList<TreeNode>();
                for(int i = 0; i < treeNum; i++) {
                    this.trees.add(new TreeNode(i, new Node(Node.ROOT_INDEX), 1d));
                }
            }
            if(this.isGBDT) {
                if(isContinuousEnabled) {
                    TreeModel existingModel;
                    try {
                        Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
                        existingModel = (TreeModel) CommonUtils.loadModel(modelConfig, modelPath,
                                ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource()));
                        if(existingModel == null) {
                            // null means no existing model file or model file is in wrong format
                            this.trees = new CopyOnWriteArrayList<TreeNode>();
                            this.trees.add(new TreeNode(0, new Node(Node.ROOT_INDEX), 1d));// learning rate is 1 for 1st
                            LOG.info("Starting to train model from scratch and existing model is empty.");
                        } else {
                            this.trees = existingModel.getTrees();
                            this.existingTreeSize = this.trees.size();
                            // starting from existing models, first tree learning rate is current learning rate
                            this.trees.add(new TreeNode(this.existingTreeSize, new Node(Node.ROOT_INDEX),
                                    this.existingTreeSize == 0 ? 1d : this.learningRate));
                            LOG.info("Starting to train model from existing model {} with existing trees {}.",
                                    modelPath, existingTreeSize);
                        }
                    } catch (IOException e) {
                        throw new GuaguaRuntimeException(e);
                    }
                } else {
                    this.trees = new CopyOnWriteArrayList<TreeNode>();
                    // for GBDT, initialize the first tree. trees are trained sequentially,first tree learning rate is 1
                    this.trees.add(new TreeNode(0, new Node(Node.ROOT_INDEX), 1.0d));
                }
            }
        } else {
            // recover all states once master is fail over
            LOG.info("Recover master status from checkpoint file {}", this.checkpointOutput);
            recoverMasterStatus(sourceType);
        }
    }

    private void recoverMasterStatus(SourceType sourceType) {
        FSDataInputStream stream = null;
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        try {
            stream = fs.open(this.checkpointOutput);
            int treeSize = stream.readInt();
            this.trees = new CopyOnWriteArrayList<TreeNode>();
            for(int i = 0; i < treeSize; i++) {
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(stream);
                this.trees.add(treeNode);
            }

            int queueSize = stream.readInt();
            for(int i = 0; i < queueSize; i++) {
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(stream);
                this.toDoQueue.offer(treeNode);
            }

            if(this.isLeafWise && this.toSplitQueue != null) {
                queueSize = stream.readInt();
                for(int i = 0; i < queueSize; i++) {
                    TreeNode treeNode = new TreeNode();
                    treeNode.readFields(stream);
                    this.toSplitQueue.offer(treeNode);
                }
            }

            this.cpMasterParams = new DTMasterParams();
            this.cpMasterParams.readFields(stream);
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        } finally {
            org.apache.commons.io.IOUtils.closeQuietly(stream);
        }
    }

}
