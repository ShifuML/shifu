package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.container.obj.ColumnRawStatsResult;
import ml.shifu.shifu.di.spi.ColumnRawStatsCalculator;
import ml.shifu.shifu.container.RawValueObject;

import java.util.List;

public class ColumnRawStatsService {

    private ColumnRawStatsCalculator screener;

    @Inject
    public ColumnRawStatsService(ColumnRawStatsCalculator screener) {
        this.screener = screener;
    }

    public ColumnRawStatsResult getResult(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {
        return screener.calculate(rvoList, posTags, negTags);
    }

}
