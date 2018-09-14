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
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethodType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 5/20/16.
 */
public class WoeZscoreLocalTransformCreator extends WoeLocalTransformCreator {

    @SuppressWarnings("unused")
    private static final Logger LOG = LoggerFactory.getLogger(WoeZscoreLocalTransformCreator.class);

    private boolean isWeightedNorm;

    public WoeZscoreLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isWeightedNorm) {
        super(modelConfig, columnConfigList);
        this.isWeightedNorm = isWeightedNorm;
    }

    public WoeZscoreLocalTransformCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isConcise, boolean isWeightedNorm) {
        super(modelConfig, columnConfigList, isConcise);
        this.isWeightedNorm = isWeightedNorm;
    }

    /**
     * Create @DerivedField for numerical variable
     *
     * @param config - ColumnConfig for numerical variable
     * @param cutoff - cutoff of normalization
     * @return DerivedField for variable
     */
    @Override
    protected List<DerivedField> createNumericalDerivedField(ColumnConfig config, double cutoff, ModelNormalizeConf.NormType normType) {
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();

        DerivedField derivedField = super.createNumericalDerivedField(config, cutoff, ModelNormalizeConf.NormType.WOE).get(0);
        derivedFields.add(derivedField);

        double[] meanAndStdDev = Normalizer.calculateWoeMeanAndStdDev(config, isWeightedNorm);

        // added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(meanAndStdDev[0] - meanAndStdDev[1] * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(meanAndStdDev[0] + meanAndStdDev[1] * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(derivedField.getName().getValue()))
                .withLinearNorms(from, to).withMapMissingTo(0.0)
                .withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        // derived field name is consisted of FieldName and "_zscl"
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
                .withName(FieldName.create(genPmmlColumnName(CommonUtils.getSimpleColumnName(config.getColumnName()), normType)))
                .withExpression(normContinuous));

        return derivedFields;
    }

    /**
     * Create @DerivedField for categorical variable
     *
     * @param config - ColumnConfig for categorical variable
     * @param cutoff - cutoff for normalization
     * @return DerivedField for variable
     */
    @Override
    protected List<DerivedField> createCategoricalDerivedField(ColumnConfig config, double cutoff, ModelNormalizeConf.NormType normType) {
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();

        DerivedField derivedField = super.createCategoricalDerivedField(config, cutoff, ModelNormalizeConf.NormType.WOE).get(0);
        derivedFields.add(derivedField);

        double[] meanAndStdDev = Normalizer.calculateWoeMeanAndStdDev(config, isWeightedNorm);

        // added capping logic to linearNorm
        LinearNorm from = new LinearNorm().withOrig(meanAndStdDev[0] - meanAndStdDev[1] * cutoff).withNorm(-cutoff);
        LinearNorm to = new LinearNorm().withOrig(meanAndStdDev[0] + meanAndStdDev[1] * cutoff).withNorm(cutoff);
        NormContinuous normContinuous = new NormContinuous(FieldName.create(derivedField.getName().getValue()))
                .withLinearNorms(from, to).withMapMissingTo(0.0)
                .withOutliers(OutlierTreatmentMethodType.AS_EXTREME_VALUES);

        // derived field name is consisted of FieldName and "_zscl"
        derivedFields.add(new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE)
                .withName(FieldName.create(genPmmlColumnName(CommonUtils.getSimpleColumnName(config.getColumnName()), normType)))
                .withExpression(normContinuous));

        return derivedFields;
    }

}
