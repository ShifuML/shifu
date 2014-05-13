package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;

import java.util.List;

public interface ColumnCatBinningCalculator {

    public ColumnBinningResult calculate(List<CategoricalValueObject> cvoList);

}
