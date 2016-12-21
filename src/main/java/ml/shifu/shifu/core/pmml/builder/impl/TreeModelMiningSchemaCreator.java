package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;

import java.util.List;

public class TreeModelMiningSchemaCreator extends AbstractPmmlElementCreator<MiningSchema> {

    public TreeModelMiningSchemaCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public TreeModelMiningSchemaCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    @Override
    public MiningSchema build() {
        MiningSchema miningSchema = new MiningSchema();

        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isFinalSelect() || columnConfig.isTarget()) {
                MiningField miningField = new MiningField();

                miningField.setName(FieldName.create(columnConfig.getColumnName()));
                miningField.setOptype(getOptype(columnConfig));
                if(columnConfig.isNumerical()) {
                    miningField.setMissingValueReplacement(String.valueOf(columnConfig.getColumnStats().getMean()));
                } else {
                    miningField.setMissingValueReplacement("");
                }
                if(columnConfig.isFinalSelect()) {
                    miningField.setUsageType(FieldUsageType.ACTIVE);
                } else if(columnConfig.isTarget()) {
                    miningField.setUsageType(FieldUsageType.TARGET);
                }

                miningSchema.withMiningFields(miningField);
            }
        }
        return miningSchema;
    }
}
