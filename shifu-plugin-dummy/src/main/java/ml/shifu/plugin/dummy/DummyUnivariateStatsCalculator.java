package ml.shifu.plugin.dummy;

import ml.shifu.core.di.spi.UnivariateStatsCalculator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Counts;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;

public class DummyUnivariateStatsCalculator implements UnivariateStatsCalculator {

    public UnivariateStats calculate(DataField field, List<? extends Object> values, Params params) {
        // Create a new UnivariateStats Object
        UnivariateStats univariateStats = new UnivariateStats();

        // Set fieldName
        univariateStats.setField(field.getName());

        // Create a new Counts Object
        Counts counts = new Counts();

        // Blindly set the totalFreq as 101
        counts.setTotalFreq(101);

        // Add the Counts to UnivariateStats
        univariateStats.setCounts(counts);

        // return the result
        return univariateStats;
    }
}
