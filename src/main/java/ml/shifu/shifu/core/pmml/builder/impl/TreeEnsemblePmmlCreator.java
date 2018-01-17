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
import ml.shifu.shifu.core.dtrain.dt.TreeNode;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Segment;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.Target;
import org.dmg.pmml.TargetValue;
import org.dmg.pmml.Targets;
import org.dmg.pmml.True;
import org.encog.ml.BasicML;

public class TreeEnsemblePmmlCreator extends AbstractPmmlElementCreator<MiningModel> {

    public TreeEnsemblePmmlCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public MiningModel build(BasicML basicML) {
        return null;
    }

    public MiningModel convert(IndependentTreeModel treeModel) {
        MiningModel gbt = new MiningModel();
        MiningSchema miningSchema = new TreeModelMiningSchemaCreator(this.modelConfig, this.columnConfigList)
                .build(null);
        gbt.setMiningSchema(miningSchema);
        if(treeModel.isClassification()) {
            gbt.setFunctionName(MiningFunctionType.fromValue("classification"));
        } else {
            gbt.setFunctionName(MiningFunctionType.fromValue("regression"));
        }
        gbt.setTargets(createTargets(this.modelConfig));

        Segmentation seg = new Segmentation();
        gbt.setSegmentation(seg);
        seg.setMultipleModelMethod(MultipleModelMethodType.fromValue("weightedAverage"));
        List<Segment> list = seg.getSegments();
        int idCount = 0;
        // such case we only support treeModel is one element list
        if(treeModel.getTrees().size() != 1) {
            throw new RuntimeException("Bagging model cannot be supported in PMML generation.");
        }
        for(TreeNode tn: treeModel.getTrees().get(0)) {
            TreeNodePmmlElementCreator tnec = new TreeNodePmmlElementCreator(this.modelConfig, this.columnConfigList,
                    treeModel);
            org.dmg.pmml.Node root = tnec.convert(tn.getNode());
            TreeModelPmmlElementCreator tmec = new TreeModelPmmlElementCreator(this.modelConfig, this.columnConfigList);
            org.dmg.pmml.TreeModel tm = tmec.convert(treeModel, root);
            tm.setModelName(String.valueOf(idCount));
            Segment segment = new Segment();
            if(treeModel.isGBDT()) {
                segment.setWeight(treeModel.getWeights().get(0).get(idCount) * treeModel.getTrees().size());
            } else {
                segment.setWeight(treeModel.getWeights().get(0).get(idCount));
            }
            segment.setId("Segement" + String.valueOf(idCount));
            idCount++;
            segment.setPredicate(new True());
            segment.setModel(tm);
            list.add(segment);
        }
        return gbt;

    }

    protected Targets createTargets(ModelConfig modelConfig) {
        Targets targets = new Targets();

        Target target = new Target();

        target.setOptype(OpType.CATEGORICAL);
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

        target.withTargetValues(targetValueList);

        targets.withTargets(target);

        return targets;
    }
}
