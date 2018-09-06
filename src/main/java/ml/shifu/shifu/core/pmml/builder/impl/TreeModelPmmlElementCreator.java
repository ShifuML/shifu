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

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Target;
import org.dmg.pmml.TargetValue;
import org.dmg.pmml.Targets;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel.MissingValueStrategy;
import org.encog.ml.BasicML;

public class TreeModelPmmlElementCreator extends AbstractPmmlElementCreator<Model> {
    
    public TreeModelPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }


    public Model build(BasicML basicML) {
        return null;
    }

    public org.dmg.pmml.tree.TreeModel convert(IndependentTreeModel model, Node root) {
        org.dmg.pmml.tree.TreeModel pmmlTreeModel = new org.dmg.pmml.tree.TreeModel();
        pmmlTreeModel.setMiningSchema(new TreeModelMiningSchemaCreator(this.modelConfig, this.columnConfigList).build(null));
        //pmmlTreeModel.setModelStats(new ModelStatsCreator(this.modelConfig, this.columnConfigList).build());
        pmmlTreeModel.setTargets(createTargets(this.modelConfig));
        pmmlTreeModel.setMissingValueStrategy(MissingValueStrategy.fromValue("none"));
        pmmlTreeModel.setSplitCharacteristic(org.dmg.pmml.tree.TreeModel.SplitCharacteristic.fromValue("binarySplit"));
        pmmlTreeModel.setModelName(String.valueOf(root.getId()));
        pmmlTreeModel.setNode(root);
        if(model.isClassification()) {
           pmmlTreeModel.setMiningFunction(MiningFunction.fromValue("classification")); 
        } else {
           pmmlTreeModel.setMiningFunction(MiningFunction.fromValue("regression"));
        }
        return pmmlTreeModel;
        
    }

    protected Targets createTargets(ModelConfig modelConfig) {
        Targets targets = new Targets();

        Target target = new Target();

        target.setOpType(OpType.CATEGORICAL);
        target.setField(new FieldName(modelConfig.getTargetColumnName()));

        List<TargetValue> targetValueList = new ArrayList<TargetValue>();

        for(String posTagValue: modelConfig.getPosTags()) {
            TargetValue pos = new TargetValue();
            pos.setValue(posTagValue);
            pos.setDisplayValue("Positive");

            targetValueList.add(pos);
        }

        for(String negTagValue: modelConfig.getNegTags()) {
            TargetValue neg = new TargetValue();
            neg.setValue(negTagValue);
            neg.setDisplayValue("Negative");

            targetValueList.add(neg);
        }

        target.addTargetValues(targetValueList.toArray(new TargetValue[targetValueList.size()]));

        targets.addTargets(target);

        return targets;
    }
}
