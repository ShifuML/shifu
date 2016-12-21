package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.Node;
import org.dmg.pmml.Targets;

import org.dmg.pmml.Model;
import org.dmg.pmml.Target;
import org.dmg.pmml.TargetValue;
import org.dmg.pmml.OpType;
import org.dmg.pmml.MissingValueStrategyType;

import java.util.List;
import java.util.ArrayList;

public class TreeModelPmmlElementCreator extends AbstractPmmlElementCreator<Model> {
    
    public TreeModelPmmlElementCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }


    public Model build() {
        return null;
    }

    public org.dmg.pmml.TreeModel convert(IndependentTreeModel model, Node root) {
        org.dmg.pmml.TreeModel pmmlTreeModel = new org.dmg.pmml.TreeModel();
        pmmlTreeModel.setMiningSchema(new TreeModelMiningSchemaCreator(this.modelConfig, this.columnConfigList).build());
        //pmmlTreeModel.setModelStats(new ModelStatsCreator(this.modelConfig, this.columnConfigList).build());
        pmmlTreeModel.setTargets(createTargets(this.modelConfig));
        pmmlTreeModel.setMissingValueStrategy(MissingValueStrategyType.fromValue("none"));
        pmmlTreeModel.setSplitCharacteristic(org.dmg.pmml.TreeModel.SplitCharacteristic.fromValue("binarySplit"));
        pmmlTreeModel.setModelName(String.valueOf(root.getId()));
        pmmlTreeModel.setNode(root);
        if(model.isClassification()) {
           pmmlTreeModel.setFunctionName(MiningFunctionType.fromValue("classification")); 
        } else {
           pmmlTreeModel.setFunctionName(MiningFunctionType.fromValue("regression"));
        }
        return pmmlTreeModel;
        
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
