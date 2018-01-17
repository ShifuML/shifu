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
import java.util.List;
import java.util.Map;

import ml.shifu.guagua.io.Bytable;

/**
 * A binary tree node.
 * 
 * <p>
 * {@link #left} and {@link #right} are children for node. Other attributes are attached as fields. {@link #predict} are
 * node predict info with predict value and classification value. To predict a node is to find a lef node and get its
 * predict.
 * 
 * <p>
 * A tree can be set as a only root node and started with id 1, 2, 3 ...
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Node implements Bytable {

    public static class NodeStats {

        /**
         * Impurity value for such node, such value can be computed from different {@link Impurity} like {@link Entropy}
         * , {@link Variance}.
         */
        private double impurity;

        /**
         * Predict value for left child, null if leaf.
         */
        private Predict leftPredict;

        /**
         * Left impurity value, 0 if leaf.
         */
        private double leftImpurity;

        /**
         * Predict value for right child, null if leaf.
         */
        private Predict rightPredict;

        /**
         * Impurity for right node.
         */
        private double rightImpurity;

        /**
         * 'isLeaf' is used to set a flag not to extend this tree.
         */
        private boolean isLeaf;

        /**
         * Ratio of # of weighted instances in such node over # of all weighted instances
         */
        private double wgtCntRatio;

        /**
         * @return the impurity
         */
        public double getImpurity() {
            return impurity;
        }

        /**
         * @return the leftPredict
         */
        public Predict getLeftPredict() {
            return leftPredict;
        }

        /**
         * @return the leftImpurity
         */
        public double getLeftImpurity() {
            return leftImpurity;
        }

        /**
         * @return the rightPredict
         */
        public Predict getRightPredict() {
            return rightPredict;
        }

        /**
         * @return the rightImpurity
         */
        public double getRightImpurity() {
            return rightImpurity;
        }

        /**
         * @param impurity
         *            the impurity to set
         */
        public void setImpurity(double impurity) {
            this.impurity = impurity;
        }

        /**
         * @param leftPredict
         *            the leftPredict to set
         */
        public void setLeftPredict(Predict leftPredict) {
            this.leftPredict = leftPredict;
        }

        /**
         * @param leftImpurity
         *            the leftImpurity to set
         */
        public void setLeftImpurity(double leftImpurity) {
            this.leftImpurity = leftImpurity;
        }

        /**
         * @param rightPredict
         *            the rightPredict to set
         */
        public void setRightPredict(Predict rightPredict) {
            this.rightPredict = rightPredict;
        }

        /**
         * @param rightImpurity
         *            the rightImpurity to set
         */
        public void setRightImpurity(double rightImpurity) {
            this.rightImpurity = rightImpurity;
        }

        /**
         * @return the isLeaf
         */
        public boolean isLeaf() {
            return isLeaf;
        }

        /**
         * @param isLeaf
         *            the isLeaf to set
         */
        public void setLeaf(boolean isLeaf) {
            this.isLeaf = isLeaf;
        }

        /**
         * @return the wgtCntRatio
         */
        public double getWgtCntRatio() {
            return wgtCntRatio;
        }

        /**
         * @param wgtCntRatio
         *            the wgtCntRatio to set
         */
        public void setWgtCntRatio(double wgtCntRatio) {
            this.wgtCntRatio = wgtCntRatio;
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#toString()
         */
        @Override
        public String toString() {
            return "NodeStats [impurity=" + impurity + ", leftPredict=" + leftPredict + ", leftImpurity="
                    + leftImpurity + ", rightPredict=" + rightPredict + ", rightImpurity=" + rightImpurity
                    + ", isLeaf=" + isLeaf + ", wgtCntRatio=" + wgtCntRatio + "]";
        }

    }

    /**
     * Node id, start from 1, 2, 3 ...
     */
    private int id;

    /**
     * Feature split for such node, if leaf node means no split. Node is split by numeric feature or categorical
     * feature. Please check {@link Split} for details.
     */
    private Split split;

    /**
     * Left child, if leaf, left is null.
     */
    private Node left;

    /**
     * Right child, if leaf, right is null.
     */
    private Node right;

    /**
     * Predict value and probability for such node which is collected from workers.
     */
    private Predict predict;

    /**
     * Node stats like gain and impurity used for split, no need in execution
     */
    private NodeStats nodeStats;

    /**
     * Gain for such node, such value can be computed from different {@link Impurity} like {@link Entropy},
     * {@link Variance}. Gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity.
     * 
     * <p>
     * Set gain to float and in fact in disk it is set to float to save disk, here set to float type to save memory
     * also.
     */
    private float gain;

    /**
     * Current weighted node count is mostly used for feature importance, by gain * wgtCount to compute contribution per
     * each node.
     */
    private double wgtCnt;

    /**
     * Default root index is 1. Others are 2, 3, 4, 5 ...
     */
    public static final int ROOT_INDEX = 1;

    /**
     * If node with such index, means such node is invalid.
     */
    public static final int INVALID_INDEX = -1;

    public Node() {
        this(ROOT_INDEX);
    }

    public Node(int id) {
        this.id = id;
    }

    public Node(int id, Node left, Node right) {
        this.id = id;
        this.left = left;
        this.right = right;
    }

    public Node(int id, Predict predict, double impurity, boolean isLeaf) {
        this.id = id;
        this.predict = predict;
        this.nodeStats = new NodeStats();
        this.nodeStats.setImpurity(impurity);
        this.nodeStats.setLeaf(isLeaf);
    }

    /**
     * @return the id
     */
    public int getId() {
        return id;
    }

    /**
     * @return the left
     */
    public Node getLeft() {
        return left;
    }

    /**
     * @return the right
     */
    public Node getRight() {
        return right;
    }

    /**
     * @return the predict
     */
    public Predict getPredict() {
        return predict;
    }

    /**
     * @return the gain
     */
    public double getGain() {
        return this.gain;
    }

    /**
     * @return the impurity
     */
    public double getImpurity() {
        return nodeStats.getImpurity();
    }

    /**
     * @return the leftPredict
     */
    public Predict getLeftPredict() {
        return nodeStats.getLeftPredict();
    }

    /**
     * @return the leftImpurity
     */
    public double getLeftImpurity() {
        return nodeStats.getLeftImpurity();
    }

    /**
     * @return the rightPredict
     */
    public Predict getRightPredict() {
        return nodeStats.getRightPredict();
    }

    /**
     * @return the rightImpurity
     */
    public double getRightImpurity() {
        return nodeStats.getRightImpurity();
    }

    /**
     * @param id
     *            the id to set
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * @param left
     *            the left to set
     */
    public void setLeft(Node left) {
        this.left = left;
    }

    /**
     * @param right
     *            the right to set
     */
    public void setRight(Node right) {
        this.right = right;
    }

    /**
     * @param predict
     *            the predict to set
     */
    public void setPredict(Predict predict) {
        this.predict = predict;
    }

    /**
     * @param gain
     *            the gain to set
     */
    public void setGain(double gain) {
        // cast to float is ok to save memory
        this.gain = (float) gain;
    }

    /**
     * @param impurity
     *            the impurity to set
     */
    public void setImpurity(double impurity) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setImpurity(impurity);
    }

    /**
     * @param leftPredict
     *            the leftPredict to set
     */
    public void setLeftPredict(Predict leftPredict) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setLeftPredict(leftPredict);
    }

    /**
     * @param leftImpurity
     *            the leftImpurity to set
     */
    public void setLeftImpurity(double leftImpurity) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setLeftImpurity(leftImpurity);
    }

    /**
     * @param rightPredict
     *            the rightPredict to set
     */
    public void setRightPredict(Predict rightPredict) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setRightPredict(rightPredict);
    }

    /**
     * @param rightImpurity
     *            the rightImpurity to set
     */
    public void setRightImpurity(double rightImpurity) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setRightImpurity(rightImpurity);
    }

    /**
     * @return the split
     */
    public Split getSplit() {
        return split;
    }

    /**
     * @param split
     *            the split to set
     */
    public void setSplit(Split split) {
        this.split = split;
    }

    public static int indexToLevel(int nodeIndex) {
        return Integer.numberOfTrailingZeros(Integer.highestOneBit(nodeIndex)) + 1;
    }

    /**
     * @param isLeaf
     *            the isLeaf to set
     */
    public void setLeaf(boolean isLeaf) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setLeaf(isLeaf);
    }

    boolean isLeaf() {
        return this.nodeStats.isLeaf();
    }

    /**
     * Check if node is real for leaf. No matter the leaf flag, this will check whether left and right exist.
     * 
     * @return if it is real leaf node
     */
    public boolean isRealLeaf() {
        return this.left == null && this.right == null;
    }

    /**
     * According to node index and topNode, find the exact node.
     * 
     * @param topNode
     *            the top node of the tree
     * @param index
     *            the index to be searched
     * @return the node with such index, or null if not found
     */
    public static Node getNode(Node topNode, int index) {
        assert index > 0 && topNode != null && topNode.id == 1;
        if(index == 1) {
            return topNode;
        }

        int currIndex = index;
        List<Integer> walkIndexes = new ArrayList<Integer>(16);
        while(currIndex > 1) {
            walkIndexes.add(currIndex);
            currIndex /= 2;
        }

        // reverse walk through
        Node result = topNode;
        for(int i = 0; i < walkIndexes.size(); i++) {
            int searchIndex = walkIndexes.get(walkIndexes.size() - 1 - i);

            if(searchIndex % 2 == 0) {
                result = result.getLeft();
            } else {
                result = result.getRight();
            }

            if(searchIndex == index) {
                return result;
            }
        }
        return null;
    }

    /**
     * Left index according to current id.
     * 
     * @param id
     *            current id
     * @return left index
     */
    public static int leftIndex(int id) {
        return id << 1;
    }

    public double getWgtCntRatio() {
        return this.nodeStats.getWgtCntRatio();
    }

    public void setWgtCntRatio(double wgtCntRatio) {
        if(this.nodeStats == null) {
            this.nodeStats = new NodeStats();
        }
        this.nodeStats.setWgtCntRatio(wgtCntRatio);
    }

    /**
     * Right index according to current id.
     * 
     * @param id
     *            current id
     * @return right index
     */
    public static int rightIndex(int id) {
        return (id << 1) + 1;
    }

    /**
     * Parent index according to current id.
     * 
     * @param id
     *            current id
     * @return parent index
     */
    public static int parentIndex(int id) {
        return id >>> 1;
    }

    /**
     * If current node is ROOT or not
     * 
     * @param node
     *            node to be checked
     * @return true if ROOT, false if not ROOT or null
     */
    public static boolean isRootNode(Node node) {
        return node != null && node.getId() == ROOT_INDEX;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(id);

        // cast to float to save space
        out.writeFloat((float) gain);

        // change current float to double to get a better accuracy, start from tree model version 3 to use it as double
        out.writeDouble(this.wgtCnt);

        if(split == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            split.write(out);
        }

        // only store needed predict info
        boolean isRealLeaf = isRealLeaf();
        out.writeBoolean(isRealLeaf);
        if(isRealLeaf) {
            if(predict == null) {
                out.writeBoolean(false);
            } else {
                out.writeBoolean(true);
                predict.write(out);
            }
        }

        if(left == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            left.write(out);
        }

        if(right == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            right.write(out);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.id = in.readInt();

        this.gain = in.readFloat();
        // for back-forward compatibility, still need to read two floats here for wgtCntRatio
        if(IndependentTreeModel.getVersion() <= 2) {
            this.wgtCnt = in.readFloat();
        } else {
            this.wgtCnt = in.readDouble();
        }

        if(in.readBoolean()) {
            this.split = new Split();
            this.split.readFields(in);
        }

        boolean isRealLeaf = in.readBoolean();
        if(isRealLeaf) {
            if(in.readBoolean()) {
                this.predict = new Predict();
                this.predict.readFields(in);
            }
        }

        if(in.readBoolean()) {
            this.left = new Node();
            this.left.readFields(in);
        }

        if(in.readBoolean()) {
            this.right = new Node();
            this.right.readFields(in);
        }
    }

    /**
     * @return the nodeStats
     */
    public NodeStats getNodeStats() {
        return nodeStats;
    }

    /**
     * @param nodeStats
     *            the nodeStats to set
     */
    public void setNodeStats(NodeStats nodeStats) {
        this.nodeStats = nodeStats;
    }

    /**
     * @return the wgtCount
     */
    public double getWgtCnt() {
        return wgtCnt;
    }

    /**
     * @param wgtCount
     *            the wgtCount to set
     */
    public void setWgtCnt(double wgtCount) {
        this.wgtCnt = wgtCount;
    }

    @Override
    public String toString() {
        return "Node [id=" + id + ", split=" + split + ", left=" + left + ", right=" + right + ", predict=" + predict
                + ", gain=" + gain + ", impurity=" + nodeStats == null ? null : nodeStats.impurity + ", leftPredict="
                + nodeStats == null ? null : nodeStats.leftPredict + ", leftImpurity=" + nodeStats == null ? null
                : nodeStats.leftImpurity + ", rightPredict=" + nodeStats == null ? null : nodeStats.rightPredict
                        + ", rightImpurity=" + nodeStats == null ? null : nodeStats.rightImpurity + "]";
    }

    public String toTree() {
        String str = "[id=" + id + ", split=" + split + ", predict=" + predict + "]\n";
        if(this.left != null) {
            str += this.left.toTree();
        }
        if(this.right != null) {
            str += this.right.toTree();
        }
        return str;
    }

    public void remapColumnNum(Map<Integer, Integer> columnMapping) {
        if(this.split != null && columnMapping.containsKey(this.split.getColumnNum())) {
            this.split.setColumnNum(columnMapping.get(this.split.getColumnNum()));
        }

        if(this.left != null) {
            this.left.remapColumnNum(columnMapping);
        }

        if(this.right != null) {
            this.right.remapColumnNum(columnMapping);
        }
    }
}
