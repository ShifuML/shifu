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

	private static SparkConf sparkConf;

	private static JavaSparkContext sc;

	public static void initSparkContext(SparkConfiguration sConf) {
		sparkConf = new SparkConf().setMaster(sConf.get("spark.master"))
				.setAppName(sConf.get("spark.app.name"));
		sc = new JavaSparkContext(sparkConf);
	}

	public static SparkConf getSparkConf() {
		if (sparkConf == null)
			throw new RuntimeException(
					"Load spark configuration before initializing SparkContext");

		return sparkConf;
	}

	public static JavaSparkContext getSc() {
		if (sc == null)
			throw new RuntimeException(
					"Load spark configuration before initializing SparkContext");

		return sc;
	}
}
