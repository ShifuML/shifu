package ml.shifu.plugin.pig.stats;

import com.google.inject.AbstractModule;

//The module class

public class PigSimpleUnivariateStatsInjector extends AbstractModule {

    @Override
    protected void configure() {
        // bind Univariate Stats to Simple Univariate Stats Calculator
        // implementation
        bind(PigUnivariateStatsCalculator.class).to(
                PigSimpleUnivariateStatsCalculator.class);

    }

}
