/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.pmml.builder.impl;

import java.util.List;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.Node;
import ml.shifu.shifu.core.dtrain.dt.Split;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.util.CommonUtils;

import org.dmg.pmml.Array;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.dmg.pmml.True;
import org.encog.ml.BasicML;

public class TreeNodePmmlElementCreator extends AbstractPmmlElementCreator<org.dmg.pmml.Node> {

    public TreeNodePmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    private IndependentTreeModel treeModel = null;

    public TreeNodePmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            IndependentTreeModel treeModel) {
        super(modelConfig, columnConfigList);
        this.treeModel = treeModel;
    }

    public void setTreeMode(IndependentTreeModel treeModel) {
        this.treeModel = treeModel;
    }

    public org.dmg.pmml.Node build(BasicML basicML) {
        return null;
    }

    public org.dmg.pmml.Node convert(Node node) {
        org.dmg.pmml.Node pmmlNode = new org.dmg.pmml.Node();
        pmmlNode.setId(String.valueOf(node.getId()));

        pmmlNode.setDefaultChild(null);
        pmmlNode.setPredicate(new True());
        pmmlNode.setEmbeddedModel(null);

        List<org.dmg.pmml.Node> childList = pmmlNode.getNodes();
        org.dmg.pmml.Node left = convert(node.getLeft(), true, node.getSplit());
        childList.add(left);
        org.dmg.pmml.Node right = convert(node.getRight(), false, node.getSplit());
        childList.add(right);

        return pmmlNode;
    }

    public org.dmg.pmml.Node convert(Node node, boolean isLeft, Split split) {
        org.dmg.pmml.Node pmmlNode = new org.dmg.pmml.Node();
        pmmlNode.setId(String.valueOf(node.getId()));
        if(node.getPredict() != null) {
            pmmlNode.setScore(String.valueOf(treeModel.isClassification() ? node.getPredict().getClassValue() : node
                    .getPredict().getPredict()));
        }
        pmmlNode.setDefaultChild(null);
        Predicate predicate = null;
        ColumnConfig columnConfig = this.columnConfigList.get(split.getColumnNum());
        if(columnConfig.isNumerical()) {
            SimplePredicate p = new SimplePredicate();
            p.setValue(String.valueOf(split.getThreshold()));
            // TODO, how to support segment variable in tree model, here should be changed
            p.setField(new FieldName(CommonUtils.getSimpleColumnName(columnConfig.getColumnName())));
            if(isLeft) {
                p.setOperator(SimplePredicate.Operator.fromValue("lessThan"));
            } else {
                p.setOperator(SimplePredicate.Operator.fromValue("greaterOrEqual"));
            }
            predicate = p;
        } else if(columnConfig.isCategorical()) {
            SimpleSetPredicate p = new SimpleSetPredicate();
            Set<Short> childCategories = split.getLeftOrRightCategories();
            // TODO, how to support segment variable in tree model, here should be changed
            p.setField(new FieldName(CommonUtils.getSimpleColumnName(columnConfig.getColumnName())));
            StringBuilder arrayStr = new StringBuilder();
            List<String> valueList = treeModel.getCategoricalColumnNameNames().get(columnConfig.getColumnNum());
            for(Short sh: childCategories) {
                if(sh >= valueList.size()) {
                    arrayStr.append(" \"\"");
                    continue;
                }
                String s = valueList.get(sh);
                arrayStr.append(" ");
                if(s.contains("\"")) {
                    String tmp = s.replaceAll("\"", "\\\\\\\"");
                    if(s.contains(" ")) {
                        arrayStr.append("\"");
                        arrayStr.append(tmp);
                        arrayStr.append("\"");
                    } else {
                        arrayStr.append(tmp);
                    }
                } else {
                    if(s.contains(" ")) {
                        arrayStr.append("\"");
                        arrayStr.append(s);
                        arrayStr.append("\"");
                    } else {
                        arrayStr.append(s);
                    }
                }
            }
            Array array = new Array(arrayStr.toString().trim(), Array.Type.fromValue("string"));
            p.setArray(array);
            if(isLeft) {
                if(split.isLeft()) {
                    p.setBooleanOperator(SimpleSetPredicate.BooleanOperator.fromValue("isIn"));
                } else {
                    p.setBooleanOperator(SimpleSetPredicate.BooleanOperator.fromValue("isNotIn"));
                }
            } else {
                if(split.isLeft()) {
                    p.setBooleanOperator(SimpleSetPredicate.BooleanOperator.fromValue("isNotIn"));
                } else {
                    p.setBooleanOperator(SimpleSetPredicate.BooleanOperator.fromValue("isIn"));
                }
            }
            predicate = p;
        }
        pmmlNode.setPredicate(predicate);
        if(node.getSplit() == null || node.isRealLeaf()) {
            return pmmlNode;
        }

        List<org.dmg.pmml.Node> childList = pmmlNode.getNodes();
        org.dmg.pmml.Node left = convert(node.getLeft(), true, node.getSplit());
        org.dmg.pmml.Node right = convert(node.getRight(), false, node.getSplit());
        childList.add(left);
        childList.add(right);
        return pmmlNode;
    }

}
