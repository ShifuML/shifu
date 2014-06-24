package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;

public interface UnivariateStatsCalculator {

    public UnivariateStats calculate(DataField field, List<? extends Object> values, Params params);

}
