package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.Normalizer;
import org.dmg.pmml.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhanhu on 5/20/16.
 */
public class WoeZscoreLocalTransformCreator extends WoeLocalTransformCreator {

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
    protected List<DerivedField> createNumericalDerivedField(ColumnConfig config, double cutoff) {
        List<DerivedField> derivedFields = new ArrayList<DerivedField>();

        DerivedField derivedField = super.createNumericalDerivedField(config, cutoff).get(0);
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
                .withName(FieldName.create(genPmmlColumnName(config.getColumnName())))
                .withExpression(normContinuous));

        return derivedFields;
    }
}
