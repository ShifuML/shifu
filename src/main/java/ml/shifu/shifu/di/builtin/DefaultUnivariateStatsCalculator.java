package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.core.UnivariateStatsContCalculator;
import ml.shifu.shifu.core.UnivariateStatsCountsCalculator;
import ml.shifu.shifu.di.spi.UnivariateStatsCalculator;
import ml.shifu.shifu.util.CommonUtils;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;

import java.util.Map;
import java.util.List;

public class DefaultUnivariateStatsCalculator  {
    private List<String> posTags;
    private List<String> negTags;
    private Integer numBins;
  /*
    public UnivariateStats calculate(DataField field, List<RawValueObject> values, Map<String, Object> params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());

        //stats.setCounts(UnivariateStatsCountsCalculator.calculate(rvoList));


        setParams(params);

        if (field.getOptype().equals(OpType.CATEGORICAL)) {
            UnivariateStatsContCalculator.calculate(stats, CommonUtils.convertListRaw2Numerical(values, posTags, negTags), numBins);


        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {
            //stats.setNumericInfo(UnivariateStatsNumericInfoCalculator.calculate(values));
        }


        return stats;
    }

    private void setParams(Map<String, Object> params) {
        if (params.containsKey("posTags")) {
            this.posTags = (List<String>) params.get("posTags");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "posTags");
        }

        if (params.containsKey("negTags")) {
            this.negTags = (List<String>) params.get("negTags");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "negTags");
        }

        if (params.containsKey("numBins")) {
            this.numBins = (Integer) params.get("numBins");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "numBins");
        }

    }
    */
}
