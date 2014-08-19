package ml.shifu.plugin.pig.stats;

import java.util.List;

import ml.shifu.core.util.Params;

import org.dmg.pmml.DataField;
import org.dmg.pmml.UnivariateStats;

import com.google.inject.Inject;

//import com.google.inject.Inject;

public class PigStatsService {

    private PigUnivariateStatsCalculator service;

    // setter method injector
    @Inject
    public void setService(PigUnivariateStatsCalculator svc) {
        this.service = svc;
    }

    public UnivariateStats calculate(DataField field,
            List<? extends Object> values, Params params) {
        // some business logic here
        return service.calculate(field, values, params);
    }
}
