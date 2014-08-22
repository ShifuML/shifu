package ml.shifu.core.di.builtin.stats;

import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.di.spi.UnivariateStatsCalculator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;

import java.util.ArrayList;
import java.util.List;

public class SimpleUnivariateStatsCalculator implements UnivariateStatsCalculator {
    private Integer numBins;

    public UnivariateStats calculate(DataField field, List<?> values, Params params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());


        SimpleUnivariateStatsCountsCalculator univariateStatsCountsCalculator = new SimpleUnivariateStatsCountsCalculator();
        univariateStatsCountsCalculator.calculate(stats, values);


        if (field.getOptype().equals(OpType.CATEGORICAL)) {

            SimpleUnivariateStatsDiscrCalculator discrCalculator = new SimpleUnivariateStatsDiscrCalculator();
            discrCalculator.calculate(stats, values, params);

        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {

            SimpleUnivariateStatsContCalculator contCalculator = new SimpleUnivariateStatsContCalculator();
            contCalculator.calculate(stats, values, params);

        }


        return stats;
    }

    @Override
    public UnivariateStats calculateRVO(DataField field, List<?> values, Params params) {
        List<Object> data = new ArrayList<Object>();
        for(Object v : values) {
            data.add(((RawValueObject)v).getValue());
        }
        return calculate(field,data,params);
    }


}
