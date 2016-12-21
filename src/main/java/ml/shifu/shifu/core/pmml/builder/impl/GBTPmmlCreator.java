package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.True;

import org.dmg.pmml.Model;
import org.dmg.pmml.Target;
import org.dmg.pmml.Targets;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Segment;
import org.dmg.pmml.OpType;
import org.dmg.pmml.TargetValue;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.MiningModel;

import org.encog.ml.BasicML;

import java.util.List;
import java.util.ArrayList;

public class GBTPmmlCreator extends AbstractPmmlElementCreator<MiningModel> {

    public GBTPmmlCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public MiningModel build() {
        return null;
    }

    public MiningModel convert(IndependentTreeModel treeModel) {
        MiningModel gbt = new MiningModel();
        MiningSchema miningSchema = new TreeModelMiningSchemaCreator(this.modelConfig, this.columnConfigList).build();
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
        for(TreeNode tn : treeModel.getTrees()) {
           TreeNodePmmlElementCreator tnec = new TreeNodePmmlElementCreator(this.modelConfig, this.columnConfigList, treeModel);
           org.dmg.pmml.Node root = tnec.convert(tn.getNode());
           TreeModelPmmlElementCreator tmec = new TreeModelPmmlElementCreator(this.modelConfig, this.columnConfigList);
           org.dmg.pmml.TreeModel tm = tmec.convert(treeModel, root);
           tm.setModelName(String.valueOf(idCount));
           Segment segment = new Segment();
           segment.setWeight(treeModel.getWeights().get(idCount) * treeModel.getTrees().size());
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
