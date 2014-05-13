package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;

import java.util.List;

public interface ColumnNumBinningCalculator {

    /**
     *
     * voList is unsorted
     * voList is filtered, only valid data(tag in either posTags or negTags)
     *
     *
     * @param nvoList
     * @param maxNumBins
     * @return
     */

    public ColumnBinningResult calculate(List<NumericalValueObject> nvoList, int maxNumBins);

}
