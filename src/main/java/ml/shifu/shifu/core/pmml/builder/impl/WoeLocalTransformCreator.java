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

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Interval;
import org.dmg.pmml.OpType;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.NormalizationUtils;

/**
 * Created by zhanhu on 3/29/16.
 */
public class WoeLocalTransformCreator extends ZscoreLocalTransformCreator {

    public WoeLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);
    }

    public WoeLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise) {
        super(modelConfig, columnConfigList, isConcise);
    }

    /**
     * Create @DerivedField for numerical variable
     * 
     * @param config
     *            - ColumnConfig for numerical variable
     * @param cutoff
     *            - cutoff of normalization
     * @param normType
     *            - the normalization method that is used to generate DerivedField
     * @return DerivedField for variable
     */
    @Override
    protected List<DerivedField> createNumericalDerivedField(ColumnConfig config, double cutoff,
            ModelNormalizeConf.NormType normType) {
        List<Double> binWoeList = (normType.equals(ModelNormalizeConf.NormType.WOE) ? config.getBinCountWoe() : config
                .getBinWeightedWoe());
        List<Double> binBoundaryList = config.getBinBoundary();

        List<DiscretizeBin> discretizeBinList = new ArrayList<DiscretizeBin>();
        for(int i = 0; i < binBoundaryList.size(); i++) {
            DiscretizeBin discretizeBin = new DiscretizeBin();

            Interval interval = new Interval();

            if(i == 0) {
                if(binBoundaryList.size() == 1) {
                    interval.setClosure(Interval.Closure.OPEN_OPEN).setLeftMargin(Double.NEGATIVE_INFINITY)
                            .setRightMargin(Double.POSITIVE_INFINITY);
                } else {
                    interval.setClosure(Interval.Closure.OPEN_OPEN).setRightMargin(binBoundaryList.get(i + 1));
                }
            } else if(i == binBoundaryList.size() - 1) {
                interval.setClosure(Interval.Closure.CLOSED_OPEN).setLeftMargin(binBoundaryList.get(i));
            } else {
                interval.setClosure(Interval.Closure.CLOSED_OPEN).setLeftMargin(binBoundaryList.get(i))
                        .setRightMargin(binBoundaryList.get(i + 1));
            }

            discretizeBin.setInterval(interval).setBinValue(Double.toString(binWoeList.get(i)));
            discretizeBinList.add(discretizeBin);
        }

        Discretize discretize = new Discretize();
        discretize
                .setDataType(DataType.DOUBLE)
                .setField(
                        FieldName.create(NormalizationUtils.getSimpleColumnName(config, columnConfigList, segmentExpansions,
                                datasetHeaders)))
                .setMapMissingTo(Normalizer.normalize(config, null, cutoff, normType).get(0).toString())
                .setDefaultValue(Normalizer.normalize(config, null, cutoff, normType).get(0).toString())
                .addDiscretizeBins(discretizeBinList.toArray(new DiscretizeBin[discretizeBinList.size()]));

        // derived field name is consisted of FieldName and "_zscl"
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).setName(
                FieldName.create(genPmmlColumnName(NormalizationUtils.getSimpleColumnName(config.getColumnName()), normType)))
                .setExpression(discretize));
        return derivedFields;
    }
}
