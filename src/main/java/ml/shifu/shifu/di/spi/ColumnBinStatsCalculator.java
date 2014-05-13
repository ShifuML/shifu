package ml.shifu.shifu.di.spi;


import ml.shifu.shifu.container.obj.ColumnBinStatsResult;
import ml.shifu.shifu.container.obj.ColumnBinningResult;

public interface ColumnBinStatsCalculator {

    public ColumnBinStatsResult calculate(ColumnBinningResult binning);

}
