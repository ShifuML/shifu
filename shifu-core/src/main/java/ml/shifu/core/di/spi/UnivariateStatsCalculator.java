package ml.shifu.core.di.spi;

import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;

public interface UnivariateStatsCalculator {

    public UnivariateStats calculate(DataField field, List<?> values, Params params);

}
