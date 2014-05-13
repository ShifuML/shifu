package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.container.obj.ColumnBinStatsResult;
import ml.shifu.shifu.di.spi.ColumnBinStatsCalculator;
import ml.shifu.shifu.container.obj.ColumnBinningResult;

public class DummyColumnBinStatsCalculator implements ColumnBinStatsCalculator {

    public ColumnBinStatsResult calculate(ColumnBinningResult binning) {
        ColumnBinStatsResult result = new ColumnBinStatsResult();
        result.setKs(2014.0);
        result.setIv(2014.0);
        return result;
    }
}
