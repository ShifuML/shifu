package ml.shifu.shifu.di.service;

import com.google.inject.Inject;
import ml.shifu.shifu.di.spi.ColumnCatBinningCalculator;
import ml.shifu.shifu.di.spi.ColumnNumBinningCalculator;
import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ColumnBinningService {

    private static Logger log = LoggerFactory.getLogger(ColumnBinningService.class);

    private ColumnNumBinningCalculator nBinning;
    private ColumnCatBinningCalculator cBinning;

    @Inject
    public ColumnBinningService(ColumnNumBinningCalculator nBinning, ColumnCatBinningCalculator cBinning) {

        log.debug("Dependency Injected: ColumnNumBinningCalculator => " + nBinning.getClass().toString());
        log.debug("Dependency Injected: ColumnCatBinningCalculator => " + cBinning.getClass().toString());
        this.nBinning = nBinning;
        this.cBinning = cBinning;
    }

    public ColumnBinningResult getNumericalResult(List<NumericalValueObject> nvoList, int maxNumBins) {
        return nBinning.calculate(nvoList, maxNumBins);
    }

    public ColumnBinningResult getCategoricalResult(List<CategoricalValueObject> cvoList) {

        return cBinning.calculate(cvoList);

    }

}
