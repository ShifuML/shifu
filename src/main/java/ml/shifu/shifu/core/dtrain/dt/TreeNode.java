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

import ml.shifu.guagua.io.Bytable;
import ml.shifu.shifu.core.dtrain.FeatureSubsetStrategy;

/**
 * {@link TreeNode} is used to wrapper node and tree index. With tree id and node in {@link TreeNode}.
 * 
 * <p>
 * {@link #features} is for sub-sampling of such node. For feature sub-sampling, {@link FeatureSubsetStrategy} includes
 * ALL, HALF and ONETHIRD.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 * 
 * @see Node
 * @see FeatureSubsetStrategy
 */
public class TreeNode implements Bytable {

    /**
     * Tree id
     */
    private int treeId;

    /**
     * Node to be wrappered
     */
    private Node node;

    /**
     * nodeNum so far in the tree
     */
    private int nodeNum;

    /**
     * Store weighted cnt of root node (id = 1) for further computing, it is meaningless if current node is not ROOT
     * node
     */
    private double rootWgtCnt = -1;

    /**
     * Sub-sampling features which is used in tree growth.
     */
    private List<Integer> features;

    /**
     * Learning rate for current model. This is very useful for GBT, since may be first 100 trees learning rate is 0.04,
     * later it is changed to 0.03.
     */
    private double learningRate = 1d;

    public TreeNode() {
    }

    public TreeNode(int treeId, Node node) {
        this.treeId = treeId;
        this.node = node;
        this.nodeNum = 1;
        this.learningRate = 1d;
    }

    public TreeNode(int treeId, Node node, double learningRate) {
        this.treeId = treeId;
        this.node = node;
        this.nodeNum = 1;
        this.learningRate = learningRate;
    }

    public TreeNode(int treeId, Node node, int nodeNum, double learningRate) {
        this.treeId = treeId;
        this.node = node;
        this.nodeNum = nodeNum;
        this.learningRate = learningRate;
    }

    public TreeNode(int treeId, Node node, List<Integer> features, double learningRate) {
        this.treeId = treeId;
        this.node = node;
        this.features = features;
        this.learningRate = learningRate;
    }

    /**
     * @return the treeId
     */
    public int getTreeId() {
        return treeId;
    }

    /**
     * @return the node
     */
    public Node getNode() {
        return node;
    }

    /**
     * @param treeId
     *            the treeId to set
     */
    public void setTreeId(int treeId) {
        this.treeId = treeId;
    }

    /**
     * @param node
     *            the node to set
     */
    public void setNode(Node node) {
        this.node = node;
    }

    /**
     * @return the features
     */
    public List<Integer> getFeatures() {
        return features;
    }

    /**
     * @param features
     *            the features to set
     */
    public void setFeatures(List<Integer> features) {
        this.features = features;
    }

    /**
     * @return the nodeNum
     */
    public int getNodeNum() {
        return nodeNum;
    }

    /**
     * Increase node number
     */
    public void incrNodeNum() {
        nodeNum += 1;
    }

    /**
     * @param nodeNum
     *            the nodeNum to set
     */
    public void setNodeNum(int nodeNum) {
        this.nodeNum = nodeNum;
    }

    /**
     * @return the rootWgtCnt
     */
    public double getRootWgtCnt() {
        return rootWgtCnt;
    }

    /**
     * @param rootWgtCnt
     *            the rootWgtCnt to set
     */
    public void setRootWgtCnt(double rootWgtCnt) {
        this.rootWgtCnt = rootWgtCnt;
    }

    /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @param learningRate
     *            the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        this.writeWithoutFeatures(out);

        if(features == null) {
            out.writeInt(0);
        } else {
            out.writeInt(features.size());
            for(Integer index: features) {
                out.writeInt(index);
            }
        }
    }

    /**
     * This is serialization version to serialize TreeNode without sub-sampling features.
     * 
     * @param out
     *            output stream
     * @throws IOException
     *             any io exception.
     */
    public void writeWithoutFeatures(DataOutput out) throws IOException {
        out.writeInt(treeId);
        out.writeInt(nodeNum);
        this.node.write(out);
        out.writeDouble(this.learningRate);

        if(this.node.getId() == Node.ROOT_INDEX) {
            out.writeDouble(this.rootWgtCnt);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.readFieldsWithoutFeatures(in);

        int len = in.readInt();
        this.features = new ArrayList<Integer>();
        for(int i = 0; i < len; i++) {
            this.features.add(in.readInt());
        }
    }

    /**
     * This is serialization version to de-serialize TreeNode without sub-sampling features.
     * 
     * @param in
     *            input stream
     * @throws IOException
     *             any io exception.
     */
    public void readFieldsWithoutFeatures(DataInput in) throws IOException {
        this.treeId = in.readInt();
        this.nodeNum = in.readInt();
        this.node = new Node();
        this.node.readFields(in);
        this.learningRate = in.readDouble();

        if(this.node.getId() == Node.ROOT_INDEX) {
            this.rootWgtCnt = in.readDouble();
        }
    }

    /**
     * Compute tree model feature importance by accumulating gain. Check scikit-learn for details:
     * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L1056
     * 
     * @return a map with (column_id, feature_importance.)
     */
    public Map<Integer, Double> computeFeatureImportance() {
        Map<Integer, Double> importances = new HashMap<Integer, Double>();
        double rootWgtCnt = node.getWgtCnt();
        preOrder(importances, node, rootWgtCnt);
        return importances;
    }

    private void preOrder(Map<Integer, Double> importances, Node node, double rootWgtCnt) {
        if(node == null) {
            return;
        }
        computeImportance(importances, node, rootWgtCnt);
        preOrder(importances, node.getLeft(), rootWgtCnt);
        preOrder(importances, node.getRight(), rootWgtCnt);
    }

    private void computeImportance(Map<Integer, Double> importances, Node node, double rootWgtCnt) {
        if(!node.isRealLeaf()) {
            int featureId = node.getSplit().getColumnNum();
            // gain * wgtcount is count* impurity - leftCount * leftImpurity - rightCount * rightImpurity in
            // scikit-learn:
            // https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L1056
            double contribution = (node.getGain() * node.getWgtCnt()) / rootWgtCnt;
            if(!importances.containsKey(featureId)) {
                importances.put(featureId, contribution);
            } else {
                importances.put(featureId, importances.get(featureId) + contribution);
            }
        }
    }

    @Override
    public String toString() {
        return "TreeNode [treeId=" + treeId + ", node=" + node.getId() + ", features=" + features + "]";
    }

    public void remapColumnNum(Map<Integer, Integer> columnMapping) {
        if(columnMapping == null || columnMapping.size() == 0) {
            return;
        }

        this.node.remapColumnNum(columnMapping);
    }
}
