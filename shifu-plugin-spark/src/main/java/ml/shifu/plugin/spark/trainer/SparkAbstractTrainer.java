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

import java.text.DecimalFormat;
import java.util.List;
import java.util.Random;

import ml.shifu.core.di.spi.Trainer;

public abstract class SparkAbstractTrainer implements Trainer {
	protected static final String LEARNING_RATE = "LearningRate";
	protected static final String PATH_OUTPUT = "pathOutput";
	protected List<List<Object>> trainDataSet;
	protected List<List<Object>> testDataSet;
	protected static final DecimalFormat df = new DecimalFormat("0.000000");
	protected static final String TRAINERID = "trainerID";

	protected void splitDataSet(List<List<Object>> fullDataSet,
			Double splitRatio, List<List<Object>> trainDataSet,
			List<List<Object>> testDataSet) {

		Random random = new Random();
		for (List<Object> pair : fullDataSet) {
			if (random.nextDouble() < splitRatio) {
				trainDataSet.add(pair);
			} else {
				testDataSet.add(pair);
			}
		}
	}

	protected abstract void saveSparkModel(String path, Object model);
}
