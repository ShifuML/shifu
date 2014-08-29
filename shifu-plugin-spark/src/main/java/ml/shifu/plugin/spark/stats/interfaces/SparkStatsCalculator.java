package ml.shifu.plugin.spark.stats.interfaces;

import ml.shifu.core.util.Params;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
/*
 * Takes an RDD and returns the PMML Stats object- ModelStats.
 * Can apply a transformation or a combination of transformations internally.
 * Separate implementations exist for Univariate and Binomial Stats.
 */
        
public interface SparkStatsCalculator {

    ModelStats calculate(JavaSparkContext jsc, JavaRDD<String> data, PMML pmml, Params params);
}
