package ml.shifu.core.di.builtin.derivedfield;

import ml.shifu.core.di.spi.PMMLDerivedFieldCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

public class ZScorePMMLDerivedFieldCreator implements PMMLDerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {
        DerivedField derivedField = new DerivedField();
        derivedField.setName(new FieldName(dataField.getName().getValue() + "_transformed"));
        derivedField.setOptype(dataField.getOptype());
        derivedField.setDataType(dataField.getDataType());


        NumericInfo numericInfo = null;

        for (UnivariateStats stats : modelStats.getUnivariateStats()) {
            if (stats.getField().equals(dataField.getName())) {
                numericInfo = stats.getNumericInfo();
                break;
            }
        }

        if (numericInfo == null) {
            throw new RuntimeException("Missing mean and stdDev in stats!");
        }

        NormContinuous normContinuous = new NormContinuous();
        normContinuous.setField(dataField.getName());

        LinearNorm linearNorm1 = new LinearNorm();
        linearNorm1.setOrig(0);
        linearNorm1.setNorm(-numericInfo.getMean() / numericInfo.getStandardDeviation());

        LinearNorm linearNorm2 = new LinearNorm();
        linearNorm2.setOrig(numericInfo.getMean());
        linearNorm2.setNorm(0);

        normContinuous.withLinearNorms(linearNorm1, linearNorm2);

        derivedField.setExpression(normContinuous);

        return derivedField;

    }
}
