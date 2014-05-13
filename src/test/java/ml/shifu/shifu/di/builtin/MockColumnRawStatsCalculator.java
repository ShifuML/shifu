package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.ColumnRawStatsResult;
import ml.shifu.shifu.di.spi.ColumnRawStatsCalculator;

import java.util.List;

public class MockColumnRawStatsCalculator implements ColumnRawStatsCalculator {

    public ColumnRawStatsResult calculate(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {
        return new ColumnRawStatsResult();
    }


}
