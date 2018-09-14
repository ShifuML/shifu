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
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;

import org.dmg.pmml.*;

import java.util.ArrayList;
import java.util.List;

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
                    interval.withClosure(Interval.Closure.OPEN_OPEN).withLeftMargin(Double.NEGATIVE_INFINITY)
                            .withRightMargin(Double.POSITIVE_INFINITY);
                } else {
                    interval.withClosure(Interval.Closure.OPEN_OPEN).withRightMargin(binBoundaryList.get(i + 1));
                }
            } else if(i == binBoundaryList.size() - 1) {
                interval.withClosure(Interval.Closure.CLOSED_OPEN).withLeftMargin(binBoundaryList.get(i));
            } else {
                interval.withClosure(Interval.Closure.CLOSED_OPEN).withLeftMargin(binBoundaryList.get(i))
                        .withRightMargin(binBoundaryList.get(i + 1));
            }

            discretizeBin.withInterval(interval).withBinValue(Double.toString(binWoeList.get(i)));
            discretizeBinList.add(discretizeBin);
        }

        Discretize discretize = new Discretize();
        discretize
                .withDataType(DataType.DOUBLE)
                .withField(
                        FieldName.create(CommonUtils.getSimpleColumnName(config, columnConfigList, segmentExpansions,
                                datasetHeaders)))
                .withMapMissingTo(Normalizer.normalize(config, null, cutoff, normType).get(0).toString())
                .withDefaultValue(Normalizer.normalize(config, null, cutoff, normType).get(0).toString())
                .withDiscretizeBins(discretizeBinList);

        // derived field name is consisted of FieldName and "_zscl"
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE).withName(
                FieldName.create(genPmmlColumnName(CommonUtils.getSimpleColumnName(config.getColumnName()), normType)))
                .withExpression(discretize));
        return derivedFields;
    }
}
