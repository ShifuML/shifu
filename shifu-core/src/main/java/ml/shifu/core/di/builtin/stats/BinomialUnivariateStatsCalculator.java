package ml.shifu.core.di.builtin.stats;

import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.di.builtin.UnivariateStatsCountsCalculator;
import ml.shifu.core.di.spi.UnivariateStatsCalculator;
import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.UnivariateStats;

import java.util.ArrayList;
import java.util.List;

public class BinomialUnivariateStatsCalculator implements UnivariateStatsCalculator {
    private List<String> posTags;
    private List<String> negTags;
    private Integer numBins;


    public UnivariateStats calculate(DataField field, List<?> values, Params params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());

        UnivariateStatsCountsCalculator.calculate(stats, values);

        
        setParams(params);
        List<String> tags = (List<String>) params.get("tags");

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


    @Override
    public UnivariateStats calculateRVO(DataField field, List<?> values, Params params) {
        UnivariateStats stats = new UnivariateStats();
        stats.setField(field.getName());
     
        List<Object> data = new ArrayList<Object>();
        for(Object obj : values) {
            data.add(((RawValueObject)obj).getValue());
        }

        UnivariateStatsCountsCalculator.calculate(stats, data);
      
        setParams(params);
        List<String> tags = (List<String>) params.get("tags");


        if (field.getOptype().equals(OpType.CATEGORICAL)) {
            BinomialUnivariateStatsDiscrCalculator.calculate(stats, CommonUtils.convertListRaw2Categorical((List<RawValueObject>)values, posTags, negTags), null);


        } else if (field.getOptype().equals(OpType.CONTINUOUS)) {

            BinomialUnivariateStatsContCalculator contCalculator = new BinomialUnivariateStatsContCalculator();
            contCalculator.calculate(stats, CommonUtils.convertListRaw2Numerical((List<RawValueObject>)values, posTags, negTags), numBins);

        }


        return stats;
    }
    
    private void setParams(Params params) {

        this.posTags = (List<String>) params.get("posTags");


        this.negTags = (List<String>) params.get("negTags");


        this.numBins = (Integer) params.get("numBins");


    }

}
