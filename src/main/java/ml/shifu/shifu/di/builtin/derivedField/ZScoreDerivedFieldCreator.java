package ml.shifu.shifu.di.builtin.derivedField;

import ml.shifu.shifu.di.spi.DerivedFieldCreator;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;

public class ZScoreDerivedFieldCreator implements DerivedFieldCreator {

    public DerivedField create(DataField dataField, ModelStats modelStats, Params params) {
        DerivedField derivedField = new DerivedField();
        derivedField.setName(dataField.getName());
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
