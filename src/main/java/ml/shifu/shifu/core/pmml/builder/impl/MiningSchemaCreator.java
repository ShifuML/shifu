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
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.util.CommonUtils;

import org.dmg.pmml.*;
import org.dmg.pmml.MiningField.UsageType;
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

        boolean isSegExpansionMode = columnConfigList.size() > datasetHeaders.length;
        int segSize = segmentExpansions.size();
        if(basicML != null && basicML instanceof BasicFloatNetwork) {
            BasicFloatNetwork bfn = (BasicFloatNetwork) basicML;
            Set<Integer> featureSet = bfn.getFeatureSet();

            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.getColumnNum() >= datasetHeaders.length) {
                    // segment expansion column no need print in DataDictionary part, assuming columnConfigList are read
                    // in order
                    break;
                }
                if(isActiveColumn(featureSet, columnConfig)) {
                    if ( columnConfig.isTarget() ) {
                        List<MiningField> miningFields = createTargetMingFields(columnConfig);
                        miningSchema.addMiningFields(miningFields.toArray(new MiningField[miningFields.size()]));
                    } else {
                        miningSchema.addMiningFields(createActiveMingFields(columnConfig));
                    }
                } else if(isSegExpansionMode) {
                    // even current column not selected, if segment column selected, we should keep raw column
                    for(int i = 0; i < segSize; i++) {
                        int newIndex = datasetHeaders.length * (i + 1) + columnConfig.getColumnNum();
                        ColumnConfig cc = columnConfigList.get(newIndex);
                        if(cc.isFinalSelect()) {
                            // if one segment feature is selected, we should put raw column in
                            if ( columnConfig.isTarget() ) {
                                List<MiningField> miningFields = createTargetMingFields(columnConfig);
                                miningSchema.addMiningFields(miningFields.toArray(new MiningField[miningFields.size()]));
                            } else {
                                miningSchema.addMiningFields(createActiveMingFields(columnConfig));
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.getColumnNum() >= datasetHeaders.length) {
                    // segment expansion column no need print in DataDictionary part, assuming columnConfigList are read
                    // in order
                    break;
                }

                // FIXME, if no variable is selected
                if(columnConfig.isFinalSelect() || columnConfig.isTarget()) {
                    if ( columnConfig.isTarget() ) {
                        List<MiningField> miningFields = createTargetMingFields(columnConfig);
                        miningSchema.addMiningFields(miningFields.toArray(new MiningField[miningFields.size()]));
                    } else {
                        miningSchema.addMiningFields(createActiveMingFields(columnConfig));
                    }
                } else if(isSegExpansionMode) {
                    // even current column not selected, if segment column selected, we should keep raw column
                    for(int i = 0; i < segSize; i++) {
                        int newIndex = datasetHeaders.length * (i + 1) + columnConfig.getColumnNum();
                        ColumnConfig cc = columnConfigList.get(newIndex);
                        if(cc.isFinalSelect()) {
                            // if one segment feature is selected, we should put raw column in
                            if ( columnConfig.isTarget() ) {
                                List<MiningField> miningFields = createTargetMingFields(columnConfig);
                                miningSchema.addMiningFields(miningFields.toArray(new MiningField[miningFields.size()]));
                            } else {
                                miningSchema.addMiningFields(createActiveMingFields(columnConfig));
                            }
                            break;
                        }
                    }
                }
            }
        }
        return miningSchema;
    }

    private MiningField createActiveMingFields(ColumnConfig columnConfig) {
        return createMiningField(
                CommonUtils.getSimpleColumnName(columnConfig.getColumnName()),
                getOptype(columnConfig), UsageType.ACTIVE);
    }

    private List<MiningField> createTargetMingFields(ColumnConfig columnConfig) {
        List<MiningField> targetMiningFields = new ArrayList<MiningField>();
        if ( modelConfig.isClassification()
                && ModelTrainConf.MultipleClassification.NATIVE.equals(modelConfig.getTrain().getMultiClassifyMethod())) {
            for ( int i = 0; i < modelConfig.getTags().size(); i ++ ) {
                targetMiningFields.add(createMiningField(
                        CommonUtils.getSimpleColumnName(columnConfig.getColumnName()) + "_" + i,
                        getOptype(columnConfig), UsageType.TARGET));
            }
        } else {
            targetMiningFields.add(createMiningField(
                    CommonUtils.getSimpleColumnName(columnConfig.getColumnName()),
                    getOptype(columnConfig), UsageType.TARGET));
        }
        return targetMiningFields;
    }

    private MiningField createMiningField(String name, OpType opType, UsageType fieldUsageType) {
        MiningField miningField = new MiningField();
        miningField.setName(FieldName.create(name));
        miningField.setOpType(opType);
        miningField.setUsageType(fieldUsageType);
        return miningField;
    }

}
