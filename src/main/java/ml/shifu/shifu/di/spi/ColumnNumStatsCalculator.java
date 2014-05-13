package ml.shifu.shifu.di.spi;


import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnNumStatsResult;

import java.util.List;

public interface ColumnNumStatsCalculator {
    public ColumnNumStatsResult calculate(List<NumericalValueObject> nvoList);
}
