package ml.shifu.core.di.builtin;

import java.util.List;

public class DefaultUnivariateStatsCalculator {
    private List<String> posTags;
    private List<String> negTags;
    private Integer numBins;
  /*
    public UnivariateStats calculate(DataField field, List<RawValueObject> values, Map<String, Object> params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());

        //stats.setCounts(UnivariateStatsCountsCalculator.calculate(rvoList));


        setGlobalParams(params);

        if (field.getOptype().equals(OpType.CATEGORICAL)) {
            UnivariateStatsContCalculator.calculate(stats, CommonUtils.convertListRaw2Numerical(values, posTags, negTags), numBins);


        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {
            //stats.setNumericInfo(UnivariateStatsNumericInfoCalculator.calculate(values));
        }


        return stats;
    }

    private void setGlobalParams(Map<String, Object> params) {
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
