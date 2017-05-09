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
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.encog.ml.BasicML;

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
    public MiningSchema build(BasicML basicML) {
        MiningSchema miningSchema = new MiningSchema();
        if(basicML instanceof BasicFloatNetwork) {
            BasicFloatNetwork bfn = (BasicFloatNetwork) basicML;
            Set<Integer> featureSet = bfn.getFeatureSet();
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.isFinalSelect() || columnConfig.isTarget()) {
                    MiningField miningField = new MiningField();

                    miningField.setName(FieldName.create(columnConfig.getColumnName()));
                    miningField.setOptype(getOptype(columnConfig));

                    if(columnConfig.isFinalSelect() && featureSet.contains(columnConfig.getColumnNum())) {
                        miningField.setUsageType(FieldUsageType.ACTIVE);
                    } else if(columnConfig.isTarget()) {
                        miningField.setUsageType(FieldUsageType.TARGET);
                    }

                    miningSchema.withMiningFields(miningField);
                }
            }
        } else {
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
        }
        return miningSchema;
    }

}
