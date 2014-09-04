/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.plugin.spark.stats;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;

import com.google.inject.Inject;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator;

/**
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
