package ml.shifu.shifu.di.spi;


import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.ColumnRawStatsResult;

import java.util.List;

public interface ColumnRawStatsCalculator {

    public ColumnRawStatsResult calculate(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags);



}
