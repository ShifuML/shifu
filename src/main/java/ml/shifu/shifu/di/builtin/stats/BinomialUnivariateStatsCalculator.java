package ml.shifu.shifu.di.builtin.stats;

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

public class BinomialUnivariateStatsCalculator implements UnivariateStatsCalculator {
    private List<String> posTags;
    private List<String> negTags;
    private Integer numBins;
    private List<String> tags;

    public UnivariateStats calculate(DataField field, List<? extends Object> values, Params params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());

        UnivariateStatsCountsCalculator.calculate(stats, values);


        setParams((Params) params.get("globalParams"));

        this.tags = (List<String>) params.get("tags");

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
            BinomialUnivariateStatsDiscrCalculator.calculate(stats, CommonUtils.convertListRaw2Categorical(rvoList, posTags, negTags), null);


        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {

            BinomialUnivariateStatsContCalculator contCalculator = new BinomialUnivariateStatsContCalculator();
            contCalculator.calculate(stats, CommonUtils.convertListRaw2Numerical(rvoList, posTags, negTags), numBins);

        }


        return stats;
    }

    private void setParams(Params params) {

        this.posTags = (List<String>) params.get("posTags");


        this.negTags = (List<String>) params.get("negTags");


        this.numBins = (Integer) params.get("numBins");


    }
}
