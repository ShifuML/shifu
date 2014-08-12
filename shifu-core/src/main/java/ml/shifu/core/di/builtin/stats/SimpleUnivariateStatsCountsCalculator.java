package ml.shifu.core.di.builtin.stats;

import org.dmg.pmml.Counts;
import org.dmg.pmml.UnivariateStats;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SimpleUnivariateStatsCountsCalculator {

    public void calculate(UnivariateStats univariateStats, List<?> values) {
        Counts counts = new Counts();

        double totalFreq = 0;
        double missingFreq = 0;
        double invalidFreq = 0;

        Set<Object> uniqueValues = new HashSet<Object>();

        for (Object value : values) {

            if (value == null) {
                missingFreq += 1.0;
            } else if (value.toString().equals("NaN")) {
                invalidFreq += 1.0;
            }

            totalFreq += 1.0;

            uniqueValues.add(value);
        }

        counts.setCardinality(uniqueValues.size());
        counts.setInvalidFreq(invalidFreq);
        counts.setMissingFreq(missingFreq);
        counts.setTotalFreq(totalFreq);

        univariateStats.setCounts(counts);
    }

}
