package ml.shifu.core.di.builtin;

import ml.shifu.core.container.RawValueObject;
import org.dmg.pmml.Counts;
import org.dmg.pmml.UnivariateStats;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class UnivariateStatsCountsCalculator {

    public static void calculate(UnivariateStats univariateStats, List<?> values) {
        Counts counts = new Counts();

        double totalFreq = 0;
        double missingFreq = 0;
        double invalidFreq = 0;

        Set<Object> uniqueValues = new HashSet<Object>();

        for (Object valueObject : values) {
            Object value;
            if (valueObject instanceof RawValueObject) {
                value = ((RawValueObject) valueObject).getValue();
            } else {
                value = valueObject;
            }


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
