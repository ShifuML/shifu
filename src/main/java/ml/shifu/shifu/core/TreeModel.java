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
package ml.shifu.shifu.core;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.Node;
import ml.shifu.shifu.core.dtrain.dt.Split;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;
import ml.shifu.shifu.util.CommonUtils;

import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

/**
 * {@link TreeModel} is to load Random Forest or Gradient Boosted Decision Tree models.
 * 
 * <p>
 * {@link #loadFromStream(InputStream, ModelConfig, List)} can be used to read serialized models.
 * 
 * <p>
 * TODO, make trees computing in parallel
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class TreeModel extends BasicML implements MLRegression {

    private static final long serialVersionUID = 1L;

    private List<ColumnConfig> columnConfigList;

    private Map<Integer, Integer> columnMapping;

    private List<TreeNode> trees;

    private List<Double> weights;

    private int inputNode;

    private boolean isGBDT = false;

    private boolean isClassification = false;

    private String algorithm;

    private String lossStr;

    public TreeModel(List<TreeNode> trees, List<Double> weights, boolean isGBDT, List<ColumnConfig> columnConfigList) {
        this.trees = trees;
        this.weights = weights;
        assert trees != null && weights != null && trees.size() == weights.size();
        this.isGBDT = isGBDT;
        this.columnConfigList = columnConfigList;
        this.columnMapping = new HashMap<Integer, Integer>(columnConfigList.size(), 1f);
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        boolean isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        int index = 0;
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(isAfterVarSelect) {
                if(!columnConfig.isMeta() && !columnConfig.isTarget() && CommonUtils.isGoodCandidate(columnConfig)) {
                    this.columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            } else {
                if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                        && columnConfig.isFinalSelect()) {
                    this.columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            }
        }
        this.inputNode = index;
    }

    public TreeModel(List<TreeNode> trees, List<Double> weights, boolean isGBDT, List<ColumnConfig> columnConfigList,
            Map<Integer, Integer> columnMapping, boolean isClassfication, String algorithm, String lossStr) {
        this.trees = trees;
        this.weights = weights;
        assert trees != null && weights != null && trees.size() == weights.size();
        this.columnConfigList = columnConfigList;
        this.columnMapping = columnMapping;
        this.inputNode = columnMapping.size();
        this.isGBDT = isGBDT;
        this.isClassification = isClassfication;
        this.algorithm = algorithm;
        this.lossStr = lossStr;
    }

    @Override
    public final MLData compute(final MLData input) {
        double[] data = input.getData();
        double predictSum = 0d;
        double weightSum = 0d;
        double[] scores = new double[this.trees.size()];
        for(int i = 0; i < this.trees.size(); i++) {
            TreeNode treeNode = this.trees.get(i);
            Double weight = this.weights.get(i);
            weightSum += weight;
            double score = predictNode(treeNode.getNode(), data);
            scores[i] = score;
            predictSum += score * weight;
        }

        MLData result = null;
        if(this.isClassification) {
            result = new BasicMLData(scores);
        } else {
            double finalPredict;
            if(this.isGBDT) {
                finalPredict = predictSum;
            } else {
                finalPredict = predictSum / weightSum;
            }
            result = new BasicMLData(1);
            result.setData(0, finalPredict);
        }
        return result;
    }

    private double predictNode(Node topNode, double[] data) {
        Node currNode = topNode;
        Split split = currNode.getSplit();
        if(split == null || currNode.isLeaf()) {
            if(this.isClassification) {
                return currNode.getPredict().getClassValue();
            } else {
                return currNode.getPredict().getPredict();
            }
        }

        ColumnConfig columnConfig = this.columnConfigList.get(split.getColumnNum());

        Node nextNode = null;
        double value = data[this.columnMapping.get(split.getColumnNum())];
        if(columnConfig.isNumerical()) {
            if(value <= split.getThreshold()) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        } else if(columnConfig.isCategorical()) {
            // value is category index + 0.1d is to avoid 0.9999999 converted to 0
            String category = columnConfig.getBinCategory().get((int) (value + 0.1d));;
            if(split.getLeftCategories().contains(category)) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        }

        assert nextNode != null;
        return predictNode(nextNode, data);
    }

    @Override
    public int getInputCount() {
        return inputNode;
    }

    @Override
    public String toString() {
        return trees.toString();
    }

    @Override
    public void updateProperties() {
        // No need implementation
    }

    public static TreeModel loadFromStream(InputStream input, List<ColumnConfig> columnConfigList) throws IOException {
        DataInputStream dis = new DataInputStream(input);
        int len = dis.readInt();
        byte[] bytes = new byte[len];
        for(int i = 0; i < bytes.length; i++) {
            bytes[i] = dis.readByte();
        }
        String algorithm = new String(bytes, "UTF-8");
        double learningRate = dis.readDouble();
        len = dis.readInt();
        bytes = new byte[len];
        for(int i = 0; i < bytes.length; i++) {
            bytes[i] = dis.readByte();
        }
        String lossStr = new String(bytes, "UTF-8");
        boolean isClassification = dis.readBoolean();
        boolean isOneVsAll = dis.readBoolean();
        int treeNum = dis.readInt();
        List<TreeNode> trees = new ArrayList<TreeNode>(treeNum);
        List<Double> weights = new ArrayList<Double>(treeNum);
        for(int i = 0; i < treeNum; i++) {
            TreeNode treeNode = new TreeNode();
            treeNode.readFields(dis);
            trees.add(treeNode);
            if(CommonConstants.RF_ALG_NAME.equalsIgnoreCase(algorithm)) {
                weights.add(1d);
            }
            if(CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(algorithm)) {
                if(i == 0) {
                    weights.add(1d);
                } else {
                    weights.add(learningRate);
                }
            }
        }

        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnConfigList.size(), 1f);
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
        boolean isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        int index = 0;
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(!isAfterVarSelect) {
                if(!columnConfig.isMeta() && !columnConfig.isTarget() && CommonUtils.isGoodCandidate(columnConfig)) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            } else {
                if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                        && columnConfig.isFinalSelect()) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            }
        }

        return new TreeModel(trees, weights, CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(algorithm),
                columnConfigList, columnMapping, isClassification && !isOneVsAll, algorithm, lossStr);
        // if one vs all, even multiple classification, treated as regression
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.MLOutput#getOutputCount()
     */
    @Override
    public int getOutputCount() {
        return 1;
    }

    /**
     * @return the trees
     */
    public List<TreeNode> getTrees() {
        return trees;
    }

    /**
     * @param trees
     *            the trees to set
     */
    public void setTrees(List<TreeNode> trees) {
        this.trees = trees;
    }

    /**
     * @return the isGBDT
     */
    public boolean isGBDT() {
        return isGBDT;
    }

    /**
     * @param isGBDT
     *            the isGBDT to set
     */
    public void setGBDT(boolean isGBDT) {
        this.isGBDT = isGBDT;
    }

    /**
     * @return the isClassfication
     */
    public boolean isClassfication() {
        return isClassification;
    }

    /**
     * @param isClassfication
     *            the isClassfication to set
     */
    public void setClassfication(boolean isClassfication) {
        this.isClassification = isClassfication;
    }

    /**
     * @return the algorithm
     */
    public String getAlgorithm() {
        return algorithm;
    }

    /**
     * @return the lossStr
     */
    public String getLossStr() {
        return lossStr;
    }

    /**
     * @param algorithm
     *            the algorithm to set
     */
    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }

    /**
     * @param lossStr
     *            the lossStr to set
     */
    public void setLossStr(String lossStr) {
        this.lossStr = lossStr;
    }

}