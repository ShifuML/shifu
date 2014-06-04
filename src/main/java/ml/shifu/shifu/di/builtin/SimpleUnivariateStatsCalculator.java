package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.di.builtin.UnivariateStatsCountsCalculator;
import ml.shifu.shifu.di.spi.UnivariateStatsCalculator;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SimpleUnivariateStatsCalculator implements UnivariateStatsCalculator {
    private Integer numBins;

    public UnivariateStats calculate(DataField field, List<? extends Object> values, Params params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());



        SimpleUnivariateStatsCountsCalculator univariateStatsCountsCalculator = new SimpleUnivariateStatsCountsCalculator();
        univariateStatsCountsCalculator.calculate(stats, values);





        if (field.getOptype().equals(OpType.CATEGORICAL)) {
            //UnivariateStatsDiscrCalculator.calculate(stats, CommonUtils.convertListRaw2Categorical(rvoList, posTags, negTags), null);
            SimpleUnivariateStatsDiscrCalculator discrCalculator = new SimpleUnivariateStatsDiscrCalculator();
            discrCalculator.calculate(stats, values, params);

        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {

            SimpleUnivariateStatsContCalculator contCalculator = new SimpleUnivariateStatsContCalculator();
            contCalculator.calculate(stats, values, params);

        }


        return stats;
    }


}
