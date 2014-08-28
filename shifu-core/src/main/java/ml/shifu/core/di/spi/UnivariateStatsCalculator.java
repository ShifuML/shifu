package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;

public interface UnivariateStatsCalculator {

    /*
     * calculate takes a DataField, a list of NumericalValueObjects, and Params
     */
    public UnivariateStats calculate(DataField field, List<?> values, Params params);

    /*
     * calculateRVO takes a DataField, a list of RawValueObjects, and Params
     */
    public UnivariateStats calculateRVO(DataField field, List<?> values, Params params);

}
