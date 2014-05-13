package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.ColumnNumStatsCalculator;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnNumStatsResult;

import java.util.List;

public class ColumnNumStatsService {

    private ColumnNumStatsCalculator calculator;

    @Inject
    public ColumnNumStatsService(ColumnNumStatsCalculator calculator) {
        this.calculator = calculator;
    }

    public ColumnNumStatsResult getResult(List<NumericalValueObject> nvoList) {
        return calculator.calculate(nvoList);
    }

}
