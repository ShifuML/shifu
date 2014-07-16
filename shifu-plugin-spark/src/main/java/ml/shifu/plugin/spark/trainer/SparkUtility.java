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
package ml.shifu.plugin.spark.trainer;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class SparkUtility {

	private static SparkConf sparkConf = new SparkConf().setMaster("local")
			.setAppName("PMMLSparkLR");

	private static JavaSparkContext sc = new JavaSparkContext(sparkConf);

	public static SparkConf getSparkConf() {
		return sparkConf;
	}

	public static JavaSparkContext getSc() {
		return sc;
	}

}
