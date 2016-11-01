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
     * Ratio of # of weighted instances in such node over # of all weighted instances
     */
    private double wgtCntRatio;

    /**
     * Gain for such node, such value can be computed from different {@link Impurity} like {@link Entropy},
     * {@link Variance}. Gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity.
     */
    private double gain;

    /**
     * Impurity value for such node, such value can be computed from different {@link Impurity} like {@link Entropy},
     * {@link Variance}.
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

    public static final int ROOT_INDEX = 1;

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
        this.impurity = impurity;
        this.isLeaf = isLeaf;
    }

    public Node(int id, Split split, Node left, Node right, Predict predict, double gain, double impurity) {
        this.id = id;
        this.split = split;
        this.left = left;
        this.right = right;
        this.predict = predict;
        this.gain = gain;
        this.impurity = impurity;
    }

    public Node(int id, Split split, Node left, Node right, Predict predict, double gain, double impurity,
            Predict leftPredict, double leftImpurity, Predict rightPredict, double rightImpurity, boolean isLeaf) {
        this.id = id;
        this.split = split;
        this.left = left;
        this.right = right;
        this.predict = predict;
        this.gain = gain;
        this.impurity = impurity;
        this.leftPredict = leftPredict;
        this.leftImpurity = leftImpurity;
        this.rightPredict = rightPredict;
        this.rightImpurity = rightImpurity;
        this.isLeaf = isLeaf;
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
        return gain;
    }

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
        this.gain = gain;
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
        this.isLeaf = isLeaf;
    }

    boolean isLeaf() {
        return this.isLeaf;
    }

    /**
     * Check if node is real for leaf. No matter the leaf flag, this will check whether left and right exist.
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

    public static int leftIndex(int id) {
        return id << 1;
    }

    /**
     * @return the wgtCnt
     */
    public double getWgtCntRatio() {
        return wgtCntRatio;
    }

    /**
     * @param wgtCnt
     *            the wgtCnt to set
     */
    public void setWgtCntRatio(double wgtCntRatio) {
        this.wgtCntRatio = wgtCntRatio;
    }

    public static int rightIndex(int id) {
        return (id << 1) + 1;
    }

    public static int parentIndex(int id) {
        return id >>> 1;
    }

    public static boolean isRootNode(Node node) {
        return node != null && node.getId() == ROOT_INDEX;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(id);
        out.writeDouble(gain);
        out.writeDouble(impurity);
        out.writeBoolean(isLeaf);
        out.writeDouble(wgtCntRatio);

        if(split == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            split.write(out);
        }

        if(predict == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            predict.write(out);
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
        this.gain = in.readDouble();
        this.impurity = in.readDouble();
        this.isLeaf = in.readBoolean();
        this.wgtCntRatio = in.readDouble();

        if(in.readBoolean()) {
            this.split = new Split();
            this.split.readFields(in);
        }

        if(in.readBoolean()) {
            this.predict = new Predict();
            this.predict.readFields(in);
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

    @Override
    public String toString() {
        return "Node [id=" + id + ", split=" + split + ", left=" + left + ", right=" + right + ", predict=" + predict
                + ", gain=" + gain + ", impurity=" + impurity + ", leftPredict=" + leftPredict + ", leftImpurity="
                + leftImpurity + ", rightPredict=" + rightPredict + ", rightImpurity=" + rightImpurity + "]";
    }

}
