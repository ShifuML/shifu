package ml.shifu.plugin.spark.stats.interfaces;

import ml.shifu.core.util.Params;

import org.dmg.pmml.UnivariateStats;

/*
 * The most basic aggregator, which 
 * 1. keeps track of a set of values which are always grouped together
 * 2. populates a UnivariateStats object based on its internal state
 */

public interface UnitState extends java.io.Serializable {

    UnitState getNewBlank();
    void merge(UnitState state) throws Exception;
    void addData(Object value);
    public void populateUnivariateStats(UnivariateStats univariateStats, Params params);

}
