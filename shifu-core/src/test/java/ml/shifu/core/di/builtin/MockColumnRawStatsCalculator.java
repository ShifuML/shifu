package ml.shifu.core.di.builtin;

import ml.shifu.core.container.ColumnRawStatsResult;
import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.di.spi.ColumnRawStatsCalculator;

import java.util.List;

public class MockColumnRawStatsCalculator implements ColumnRawStatsCalculator {

    public ColumnRawStatsResult calculate(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {
        return new ColumnRawStatsResult();
    }


}
