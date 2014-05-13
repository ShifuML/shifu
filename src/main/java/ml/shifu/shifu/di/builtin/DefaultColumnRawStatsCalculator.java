package ml.shifu.shifu.di.builtin;


import ml.shifu.shifu.container.obj.ColumnRawStatsResult;
import ml.shifu.shifu.di.spi.ColumnRawStatsCalculator;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.util.CommonUtils;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class DefaultColumnRawStatsCalculator implements ColumnRawStatsCalculator {

    public ColumnRawStatsResult calculate(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {
        Integer cntTotal = 0;
        Integer cntValidPositive = 0;
        Integer cntValidNegative = 0;
        Integer cntIgnoredByTag = 0;
        Integer cntIsNull = 0;
        Integer cntIsNaN = 0;
        Integer cntIsNumber = 0;
        Integer cntUniqueValues;
        Set<Object> uniqueValues = new HashSet<Object>();
        ColumnRawStatsResult result = new ColumnRawStatsResult();

        for (RawValueObject rvo : rvoList) {
            cntTotal += 1;

            if (rvo.getValue() == null) {
                cntIsNull += 1;
            } else if (rvo.getValue().toString().equals("NaN")) {
                cntIsNaN += 1;
            } else {

                String raw = rvo.getValue().toString();

                if (posTags.contains(rvo.getTag())) {

                    cntValidPositive += 1;
                } else if (negTags.contains(rvo.getTag())) {
                    cntValidNegative += 1;
                } else {
                    cntIgnoredByTag += 1;
                }

                if (CommonUtils.isValidNumber(raw)) {
                    cntIsNumber += 1;
                }

            }

            uniqueValues.add(rvo.getValue());
        }
        cntUniqueValues = uniqueValues.size();

        result.setCntTotal(cntTotal);
        result.setCntValidPositive(cntValidPositive);
        result.setCntValidNegative(cntValidNegative);
        result.setCntIgnoredByTag(cntIgnoredByTag);
        result.setCntIsNumber(cntIsNumber);
        result.setCntUniqueValues(cntUniqueValues);
        result.setCntIsNull(cntIsNull);
        result.setCntIsNaN(cntIsNaN);

        return result;
    }
}
