package ml.shifu.plugin.dummy;

import ml.shifu.core.di.spi.UnivariateStatsCalculator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Counts;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;

public class DummyUnivariateStatsCalculator implements UnivariateStatsCalculator {

    public UnivariateStats calculate(DataField field, List<? extends Object> values, Params params) {
        UnivariateStats univariateStats = new UnivariateStats();
        univariateStats.setField(field.getName());

        Counts counts = new Counts();
        counts.setTotalFreq(3.1415);

        univariateStats.setCounts(counts);
        return univariateStats;
    }
}
