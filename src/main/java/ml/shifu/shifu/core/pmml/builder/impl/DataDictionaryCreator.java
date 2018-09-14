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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractPmmlElementCreator;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.encog.ml.BasicML;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Created by zhanhu on 3/29/16.
 */
public class DataDictionaryCreator extends AbstractPmmlElementCreator<DataDictionary> {

    public DataDictionaryCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public DataDictionaryCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    @Override
    public DataDictionary build(BasicML basicML) {
        DataDictionary dict = new DataDictionary();
        List<DataField> fields = new ArrayList<DataField>();

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
                if(isConcise) {
                    if(columnConfig.isFinalSelect()
                            && (CollectionUtils.isEmpty(featureSet) || featureSet.contains(columnConfig.getColumnNum()))
                            || columnConfig.isTarget()) {
                        fields.add(convertColumnToDataField(columnConfig));
                    } else if(isSegExpansionMode) {
                        // even current column not selected, if segment column selected, we should keep raw column
                        for(int i = 0; i < segSize; i++) {
                            int newIndex = datasetHeaders.length * (i + 1) + columnConfig.getColumnNum();
                            ColumnConfig cc = columnConfigList.get(newIndex);
                            if(cc.isFinalSelect()) {
                                // if one segment feature is selected, we should put raw column in
                                fields.add(convertColumnToDataField(columnConfig));
                                break;
                            }
                        }
                    }
                } else {
                    fields.add(convertColumnToDataField(columnConfig));
                }
            }
        } else {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.getColumnNum() >= datasetHeaders.length) {
                    // segment expansion column no need print in DataDictionary part, assuming columnConfigList are read
                    // in order
                    break;
                }
                if(isConcise) {
                    if(columnConfig.isFinalSelect() || columnConfig.isTarget()) {
                        fields.add(convertColumnToDataField(columnConfig));
                    } else if(isSegExpansionMode) {
                        // even current column not selected, if segment column selected, we should keep raw column
                        for(int i = 0; i < segSize; i++) {
                            int newIndex = datasetHeaders.length * (i + 1) + columnConfig.getColumnNum();
                            ColumnConfig cc = columnConfigList.get(newIndex);
                            if(cc.isFinalSelect()) {
                                // if one segment feature is selected, we should put raw column in
                                fields.add(convertColumnToDataField(columnConfig));
                                break;
                            }
                        }
                    }
                } else {
                    fields.add(convertColumnToDataField(columnConfig));
                }
            }
        }

        dict.withDataFields(fields);
        dict.withNumberOfFields(fields.size());
        return dict;
    }

    private DataField convertColumnToDataField(ColumnConfig columnConfig) {
        DataField field = new DataField();
        field.setName(FieldName.create(CommonUtils.getSimpleColumnName(columnConfig.getColumnName())));
        field.setOptype(getOptype(columnConfig));
        field.setDataType(getDataType(field.getOptype()));
        return field;
    }

}
