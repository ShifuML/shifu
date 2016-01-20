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
                    GainInfo maxGainInfoInFeature = null;
                    double[] statsArray = gainEntry.getValue();
                    if(config.isNumerical()) {
                        double count = 0d;
                        double sum = 0d;
                        double sumSquare = 0d;
                        for(int i = 0; i < statsArray.length / 3; i++) {
                            count += statsArray[i * 3];
                            sum += statsArray[i * 3 + 1];
                            sumSquare += statsArray[i * 3 + 2];
                        }

                        double impurity = 0d;
                        if(count != 0d) {
                            impurity = (sumSquare - (sum * sum) / count) / count;
                        }
                        Predict predict = new Predict(sum / count);

                        double leftCount = 0d;
                        double leftSum = 0d;
                        double leftSumSquare = 0d;
                        double rightCount = 0d;
                        double rightSum = 0d;
                        double rightSumSquare = 0d;
                        List<GainInfo> internalGainList = new ArrayList<GainInfo>();
                        for(int i = 0; i < statsArray.length / 3; i++) {
                            leftCount += statsArray[i * 3];
                            leftSum += statsArray[i * 3 + 1];
                            leftSumSquare += statsArray[i * 3 + 2];
                            rightCount = count - leftCount;
                            rightSum = sum - leftSum;
                            rightSumSquare = sumSquare - leftSumSquare;
                            double leftWeight = leftCount / count;
                            double rightWeight = rightCount / count;
                            double leftImpurity = 0d;
                            if(leftCount != 0d) {
                                leftImpurity = (leftSumSquare - (leftSum * leftSum) / leftCount) / leftCount;
                            }
                            double rightImpurity = 0d;
                            if(rightCount != 0d) {
                                rightImpurity = (rightSumSquare - (rightSum * rightSum) / rightCount) / rightCount;
                            }
                            double gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity;
                            Split split = new Split(columnNum, FeatureType.CONTINUOUS, config.getBinBoundary().get(i),
                                    null);
                            Predict leftPredict = new Predict(leftSum / leftCount);
                            Predict rightPredict = new Predict(rightSum / rightCount);
                            internalGainList.add(new GainInfo(gain, impurity, predict, leftImpurity, rightImpurity,
                                    leftPredict, rightPredict, split));
                        }
                        maxGainInfoInFeature = getGainInfoByMaxGain(internalGainList);
                    } else if(config.isCategorical()) {
                        // TODO
                    }
                    gainList.add(maxGainInfoInFeature);
                }

                GainInfo maxGainInfo = getGainInfoByMaxGain(gainList);
                doneNode.setPredict(maxGainInfo.predict);
                doneNode.setSplit(maxGainInfo.split);
                doneNode.setGain(maxGainInfo.gain);
                doneNode.setImpurity(maxGainInfo.impurity);
                doneNode.setLeftImpurity(maxGainInfo.leftImpurity);
                doneNode.setRightImpurity(maxGainInfo.rightImpurity);
                doneNode.setLeftPredict(maxGainInfo.leftPredict);
                doneNode.setRightPredict(maxGainInfo.rightPredict);

                boolean isLeaf = maxGainInfo.gain <= 0 || Node.indexToLevel(doneNode.getId()) == this.maxDepth;
                doneNode.setLeaf(isLeaf);

                if(!doneNode.isLeaf()) {
                    boolean leftChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                            || (maxGainInfo.leftImpurity == 0.0);
                    Node left = new Node(Node.leftIndex(doneNode.getId()), maxGainInfo.leftPredict,
                            maxGainInfo.leftImpurity, leftChildIsLeaf);
                    boolean rightChildIsLeaf = Node.indexToLevel(doneNode.getId()) + 1 == this.maxDepth
                            || (maxGainInfo.rightImpurity == 0.0);
                    Node right = new Node(Node.rightIndex(doneNode.getId()), maxGainInfo.rightPredict,
                            maxGainInfo.rightImpurity, rightChildIsLeaf);
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

            DTMasterParams masterParams = new DTMasterParams(trees, todoNodes);
            if(queue.isEmpty()) {
                masterParams.setHalt(true);
            }
            return masterParams;
        }
    }

    private long getStatsMem(List<Integer> subsetFeatures) {
        long statsMem = 0L;
        for(Integer columnNum: subsetFeatures) {
            ColumnConfig config = this.columnConfigList.get(columnNum);
            // TODO according to Gini, Entropy or Variance
            if(config.isNumerical()) {
                statsMem += (config.getBinBoundary().size() + 1) * 3 * 8L;
            } else if(config.isCategorical()) {
                statsMem += (config.getBinCategory().size() + 1) * 3 * 8L;
            }
        }
        return statsMem;
    }

    public GainInfo getGainInfoByMaxGain(List<GainInfo> gainList) {
        double maxGain = Double.MIN_VALUE;
        int maxGainIndex = -1;
        for(int i = 0; i < gainList.size(); i++) {
            double gain = gainList.get(i).gain;
            if(gain > maxGain) {
                maxGain = gain;
                maxGainIndex = i;
            }
        }
        return gainList.get(maxGainIndex);
    }

    private static class GainInfo {

        public GainInfo(double gain, double impurity, Predict predict, double leftImpurity, double rightImpurity,
                Predict leftPredict, Predict rightPredict, Split split) {
            this.gain = gain;
            this.impurity = impurity;
            this.predict = predict;
            this.leftImpurity = leftImpurity;
            this.rightImpurity = rightImpurity;
            this.leftPredict = leftPredict;
            this.rightPredict = rightPredict;
            this.split = split;
        }

        private double gain;

        private double impurity;

        private Predict predict;

        private double leftImpurity;

        private double rightImpurity;

        private Predict leftPredict;

        private Predict rightPredict;

        private Split split;
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

    // TODO refactor me please
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
        this.trees = new ArrayList<TreeNode>(treeNum);
        for(int i = 0; i < treeNum; i++) {
            this.trees.add(new TreeNode(i, new Node(Node.ROOT_INDEX)));
        }

        String imStr = this.modelConfig.getTrain().getParams().get("impurity").toString();
        if(imStr.equalsIgnoreCase("entropy")) {
            impurity = new Entropy();
        } else if(imStr.equalsIgnoreCase("gini")) {
            impurity = new Gini();
        } else {
            impurity = new Variance();
        }

        this.queue = new LinkedList<TreeNode>();
        // TODO recover state trees here for fail-over
    }
}
