package ml.shifu.plugin.spark.stats;

import java.util.List;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;
import ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator;

import org.apache.spark.Accumulable;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.DataField;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.dmg.pmml.UnivariateStats;

/*
 * Implementation of SparkStatsCalculator for Univariate stats.
 * Applies an accumulator (UnivariateColumnStateArray) over the RDD.
 */

public class UnivariateStatsCalculator implements
        SparkStatsCalculator {

    public ModelStats calculate(JavaSparkContext jsc, JavaRDD<String> data, PMML pmml, Params bindingParams) {
        List<DataField> dataFields= pmml.getDataDictionary().getDataFields();
        Accumulable<ColumnStateArray, String> accum= jsc.accumulable(new UnivariateColumnStateArray(dataFields, bindingParams), new SparkAccumulableWrapper());
        data.foreach(new AccumulatorApplicator(accum));
        ColumnStateArray colStateArray= accum.value();
        return  colStateArray.getModelStats();
    }
}
