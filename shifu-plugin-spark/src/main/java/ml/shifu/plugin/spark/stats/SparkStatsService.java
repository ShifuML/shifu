package ml.shifu.plugin.spark.stats;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;

import com.google.inject.Inject;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator;

/*
 * Class used exclusively for performing DI on SparkStatsCalculator
 * HAS A SparkStatsCalculator: Can be either a Univariate or Binomial StatsCalculator
 */
public class SparkStatsService {
    private SparkStatsCalculator statsCalculator;
    @Inject
    public SparkStatsService(SparkStatsCalculator statsCalculator) {
        this.statsCalculator = statsCalculator;
    }

    public ModelStats calculate(JavaSparkContext jsc, JavaRDD<String> data, PMML pmml, Params params) {

        return statsCalculator.calculate(jsc, data, pmml, params);

    }

}
