package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;

import java.util.List;

/**
 * Created by zhanhu on 3/29/16.
 */
public class MiningSchemaCreator extends AbstractPmmlElementCreator<MiningSchema> {

    public MiningSchemaCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public MiningSchemaCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
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
