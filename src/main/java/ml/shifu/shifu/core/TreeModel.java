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
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class TreeModel extends BasicML implements MLRegression {

    private static final long serialVersionUID = 1L;

    private List<ColumnConfig> columnConfigList;

    private Map<Integer, Integer> columnMapping;

    private List<TreeNode> trees;

    private int inputNode;

    public TreeModel(List<TreeNode> trees, List<ColumnConfig> columnConfigList) {
        this.trees = trees;
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

    @Override
    public final MLData compute(final MLData input) {
        double[] data = input.getData();
        double predictSum = 0d;
        for(TreeNode treeNode: trees) {
            predictSum += predictNode(treeNode.getNode(), data);
        }
        MLData result = new BasicMLData(1);
        result.setData(0, predictSum / trees.size());
        return result;
    }

    private double predictNode(Node topNode, double[] data) {
        Node currNode = topNode;
        Split split = currNode.getSplit();
        if(split == null || currNode.isLeaf()) {
            return currNode.getPredict().getPredict();
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
        int treeNum = dis.readInt();
        List<TreeNode> trees = new ArrayList<TreeNode>(treeNum);
        for(int i = 0; i < treeNum; i++) {
            TreeNode treeNode = new TreeNode();
            treeNode.readFields(dis);
            trees.add(treeNode);
        }
        return new TreeModel(trees, columnConfigList);
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
}