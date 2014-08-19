package ml.shifu.plugin.pig.stats;

import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;

//MessengerService (the contract)

public interface PigUnivariateStatsCalculator {

    public UnivariateStats calculate(DataField field,
            List<? extends Object> values, Params params);

}