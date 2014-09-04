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
package ml.shifu.plugin.spark.stats.interfaces;

import ml.shifu.core.util.Params;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
/**
 * Takes an RDD and returns the PMML Stats object- ModelStats.
 * Can apply a transformation or a combination of transformations internally.
 * Separate implementations exist for Univariate and Binomial Stats.
 */
        
public interface SparkStatsCalculator {

    ModelStats calculate(JavaSparkContext jsc, JavaRDD<String> data, PMML pmml, Params params);
}
