package ml.shifu.core.di.service;

import com.google.inject.Inject;
import ml.shifu.core.di.spi.UnivariateStatsCalculator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import java.util.List;


public class UnivariateStatsService {

    private UnivariateStatsCalculator univariateStatsCalculator;

    @Inject
    public UnivariateStatsService(UnivariateStatsCalculator univariateStatsCalculator) {
        this.univariateStatsCalculator = univariateStatsCalculator;
    }

    public UnivariateStats getUnivariateStats(DataField dataField, List<?> values, Params params) {



        return univariateStatsCalculator.calculate(dataField, values, params);

    }


}
