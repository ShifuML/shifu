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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.guagua.io.HaltBytable;

/**
 * Master parameters transferred from master to all workers in all iterations.
 * 
 * <p>
 * Every iteration, tree root nodes {@link #trees} are transferred to avoid maintain such updated trees in workers.
 * 
 * <p>
 * Every time for Random Forest, all {@link #trees} will be transfered to workers. While for GBDT, only current tree
 * will be transfered to workers. Worker recover from checkpoint trees in each iteration from worker.
 * 
 * <p>
 * {@link #tmpTrees} is transient and only for GBDT, in {@link DTOutput}, {@link #tmpTrees} is used to save model to
 * HDFS while not sent to workers.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DTMasterParams extends HaltBytable {

    /**
     * All updated trees.
     */
    private List<TreeNode> trees;

    /**
     * nodeIndexInGroup => (treeId, Node); nodeIndexInGroup is starting from 0 in each iteration.
     */
    private Map<Integer, TreeNode> todoNodes;

    /**
     * Sum of weighted training counts accumulated by workers.
     */
    private double trainCount;

    /**
     * Sum of weighted validation counts accumulated by workers.
     */
    private double validationCount;

    /**
     * Sum of train error accumulated by workers.
     */
    private double trainError;

    /**
     * Sum of validation error accumulated by workers.
     */
    private double validationError;

    /**
     * For GBDT only, in GBDT, this means move compute to next tree.
     */
    private boolean isSwitchToNextTree = false;

    /**
     * Tree depth per tree index which is used to show on each iteration.
     */
    private List<Integer> treeDepth = new ArrayList<Integer>();

    /**
     * If it is continuous running at first iteration in master
     */
    private boolean isContinuousRunningStart = false;

    /**
     * Check if it is the first tree
     */
    private boolean isFirstTree = false;

    /**
     * Tmp trees and reference from DTMaster#trees, which cannot and will not be serialized from master worker
     * iteration, only for DTOutput reference.
     */
    private List<TreeNode> tmpTrees;

    public DTMasterParams() {
    }

    public DTMasterParams(double trainCount, double trainError, double validationCount, double validationError) {
        this.trainCount = trainCount;
        this.trainError = trainError;
        this.validationCount = validationCount;
        this.validationError = validationError;
    }

    public DTMasterParams(List<TreeNode> trees, Map<Integer, TreeNode> todoNodes) {
        this.trees = trees;
        this.todoNodes = todoNodes;
    }

    /**
     * @return the trees
     */
    public List<TreeNode> getTrees() {
        return trees;
    }

    /**
     * @return the todoNodes
     */
    public Map<Integer, TreeNode> getTodoNodes() {
        return todoNodes;
    }

    /**
     * @param trees
     *            the trees to set
     */
    public void setTrees(List<TreeNode> trees) {
        this.trees = trees;
    }

    /**
     * @param todoNodes
     *            the todoNodes to set
     */
    public void setTodoNodes(Map<Integer, TreeNode> todoNodes) {
        this.todoNodes = todoNodes;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        out.writeDouble(trainCount);
        out.writeDouble(validationCount);
        out.writeDouble(trainError);
        out.writeDouble(validationError);
        out.writeBoolean(this.isSwitchToNextTree);

        assert trees != null;
        out.writeInt(trees.size());
        for(TreeNode node: trees) {
            node.writeWithoutFeatures(out);
        }

        if(todoNodes == null) {
            out.writeInt(0);
        } else {
            out.writeInt(todoNodes.size());
            for(Map.Entry<Integer, TreeNode> node: todoNodes.entrySet()) {
                out.writeInt(node.getKey());
                // for todo nodes, no left and right node, so node serialization not waste space
                node.getValue().write(out);
            }
        }
        out.writeBoolean(isContinuousRunningStart);
        out.writeBoolean(isFirstTree);
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        this.trainCount = in.readDouble();
        this.validationCount = in.readDouble();
        this.trainError = in.readDouble();
        this.validationError = in.readDouble();
        this.isSwitchToNextTree = in.readBoolean();

        int treeNum = in.readInt();
        this.trees = new ArrayList<TreeNode>(treeNum);
        for(int i = 0; i < treeNum; i++) {
            TreeNode treeNode = new TreeNode();
            treeNode.readFieldsWithoutFeatures(in);
            this.trees.add(treeNode);
        }

        int todoNodesSize = in.readInt();
        if(todoNodesSize > 0) {
            todoNodes = new HashMap<Integer, TreeNode>(todoNodesSize, 1f);
            for(int i = 0; i < todoNodesSize; i++) {
                int key = in.readInt();
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(in);
                todoNodes.put(key, treeNode);
            }
        }
        this.isContinuousRunningStart = in.readBoolean();
        this.isFirstTree = in.readBoolean();
    }

    /**
     * @return the trainCount
     */
    public double getTrainCount() {
        return trainCount;
    }

    /**
     * @return the validationCount
     */
    public double getValidationCount() {
        return validationCount;
    }

    /**
     * @param trainCount
     *            the trainCount to set
     */
    public void setTrainCount(double trainCount) {
        this.trainCount = trainCount;
    }

    /**
     * @param validationCount
     *            the validationCount to set
     */
    public void setValidationCount(double validationCount) {
        this.validationCount = validationCount;
    }

    /**
     * @return the squareError
     */
    public double getTrainError() {
        return trainError;
    }

    /**
     * @param squareError
     *            the squareError to set
     */
    public void setTrainError(double squareError) {
        this.trainError = squareError;
    }

    /**
     * @return the isSwitchToNextTree
     */
    public boolean isSwitchToNextTree() {
        return isSwitchToNextTree;
    }

    /**
     * @param isSwitchToNextTree
     *            the isSwitchToNextTree to set
     */
    public void setSwitchToNextTree(boolean isSwitchToNextTree) {
        this.isSwitchToNextTree = isSwitchToNextTree;
    }

    /**
     * @return the treeDepth
     */
    public List<Integer> getTreeDepth() {
        return treeDepth;
    }

    /**
     * @param treeDepth
     *            the treeDepth to set
     */
    public void setTreeDepth(List<Integer> treeDepth) {
        this.treeDepth = treeDepth;
    }

    /**
     * @return the validationError
     */
    public double getValidationError() {
        return validationError;
    }

    /**
     * @param validationError
     *            the validationError to set
     */
    public void setValidationError(double validationError) {
        this.validationError = validationError;
    }

    /**
     * @return the isContinuousRunningStart
     */
    public boolean isContinuousRunningStart() {
        return isContinuousRunningStart;
    }

    /**
     * @param isContinuousRunningStart
     *            the isContinuousRunningStart to set
     */
    public void setContinuousRunningStart(boolean isContinuousRunningStart) {
        this.isContinuousRunningStart = isContinuousRunningStart;
    }

    /**
     * @return the tmpTrees
     */
    public List<TreeNode> getTmpTrees() {
        return tmpTrees;
    }

    /**
     * @param tmpTrees
     *            the tmpTrees to set
     */
    public void setTmpTrees(List<TreeNode> tmpTrees) {
        this.tmpTrees = tmpTrees;
    }

    /**
     * @return the isFirstTree
     */
    public boolean isFirstTree() {
        return isFirstTree;
    }

    /**
     * @param isFirstTree
     *            the isFirstTree to set
     */
    public void setFirstTree(boolean isFirstTree) {
        this.isFirstTree = isFirstTree;
    }

}
