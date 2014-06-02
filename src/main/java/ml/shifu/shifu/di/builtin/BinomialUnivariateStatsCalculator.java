package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.core.UnivariateStatsContCalculator;
import ml.shifu.shifu.core.UnivariateStatsCountsCalculator;
import ml.shifu.shifu.core.UnivariateStatsDiscrCalculator;
import ml.shifu.shifu.di.spi.UnivariateStatsCalculator;
import ml.shifu.shifu.util.CommonUtils;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BinomialUnivariateStatsCalculator implements UnivariateStatsCalculator {
    private List<String> posTags;
    private List<String> negTags;
    private Integer numBins;
    private List<String> tags;

    public UnivariateStats calculate(DataField field, List<? extends Object> values, Map<String, Object> params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());

        UnivariateStatsCountsCalculator.calculate(stats, values);


        setParams(params);

        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        int size = values.size();
        for (int i = 0; i < size; i++) {
            Object value = values.get(i);
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(value);
            rvo.setTag(tags.get(i));

            rvoList.add(rvo);
        }



        if (field.getOptype().equals(OpType.CATEGORICAL)) {
            UnivariateStatsDiscrCalculator.calculate(stats, CommonUtils.convertListRaw2Categorical(rvoList, posTags, negTags), null);


        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {
            //stats.setNumericInfo(UnivariateStatsNumericInfoCalculator.calculate(values));
            UnivariateStatsContCalculator contCalculator = new UnivariateStatsContCalculator();
            contCalculator.calculate(stats, CommonUtils.convertListRaw2Numerical(rvoList, posTags, negTags), numBins);

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

        if (params.containsKey("tags")) {
            this.tags = (List<String>) params.get("tags");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "tags");
        }

        if (params.containsKey("numBins")) {
            this.numBins = (Integer) params.get("numBins");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "numBins");
        }

    }
}
