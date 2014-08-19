package ml.shifu.plugin.pig.stats;

import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;

import com.google.inject.Singleton;

import java.util.List;

@Singleton
public class PigSimpleUnivariateStatsCalculator implements
        PigUnivariateStatsCalculator {
    private Integer numBins;

    public UnivariateStats calculate(DataField field,
            List<? extends Object> values, Params params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());

        PigSimpleUnivariateStatsCountsCalculator univariateStatsCountsCalculator = new PigSimpleUnivariateStatsCountsCalculator();
        univariateStatsCountsCalculator.calculate(stats, values);

        if (field.getOptype().equals(OpType.CATEGORICAL)) {
            // UnivariateStatsDiscrCalculator.calculate(stats,
            // CommonUtils.convertListRaw2Categorical(rvoList, posTags,
            // negTags), null);
            PigSimpleUnivariateStatsDiscrCalculator discrCalculator = new PigSimpleUnivariateStatsDiscrCalculator();
            discrCalculator.calculate(stats, values, params);

        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {

            PigSimpleUnivariateStatsContCalculator contCalculator = new PigSimpleUnivariateStatsContCalculator();
            contCalculator.calculate(stats, values, params);

        }

        return stats;
    }

}