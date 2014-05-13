package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.ColumnBinStatsCalculator;
import ml.shifu.shifu.container.obj.ColumnBinStatsResult;
import ml.shifu.shifu.container.obj.ColumnBinningResult;


public class ColumnBinStatsService {

    ColumnBinStatsCalculator calculator;

    @Inject
    public ColumnBinStatsService(ColumnBinStatsCalculator calculator) {
         this.calculator = calculator;
    }

    public ColumnBinStatsResult getResult(ColumnBinningResult binningResult) {
          return calculator.calculate(binningResult);
    }

}
