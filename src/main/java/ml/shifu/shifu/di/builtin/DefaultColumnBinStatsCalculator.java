package ml.shifu.shifu.di.builtin;


import ml.shifu.shifu.di.spi.ColumnBinStatsCalculator;
import ml.shifu.shifu.container.obj.ColumnBinStatsResult;
import ml.shifu.shifu.container.obj.ColumnBinningResult;
import ml.shifu.shifu.core.KSIVCalculator;
import ml.shifu.shifu.core.WOECalculator;

public class DefaultColumnBinStatsCalculator implements ColumnBinStatsCalculator {

    public ColumnBinStatsResult calculate(ColumnBinningResult binning) {
        ColumnBinStatsResult stats = new ColumnBinStatsResult();
        stats.setBinWoe(WOECalculator.calculate(binning.getBinCountPos().toArray(), binning.getBinCountNeg().toArray()));

        KSIVCalculator calculator = new KSIVCalculator();
        calculator.calculateKSIV(binning.getBinCountNeg(), binning.getBinCountPos());
        stats.setIv(calculator.getIV());
        stats.setKs(calculator.getKS());

        return stats;
    }

}
