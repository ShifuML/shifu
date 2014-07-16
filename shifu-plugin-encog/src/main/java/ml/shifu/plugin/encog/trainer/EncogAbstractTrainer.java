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
package ml.shifu.plugin.encog.trainer;

import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.spi.Trainer;
import ml.shifu.core.util.PMMLUtils;

import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;

public abstract class EncogAbstractTrainer implements Trainer {
	protected static final String LEARNING_RATE = "LearningRate";
	protected static final String PATH_OUTPUT = "pathOutput";

	protected static final DecimalFormat df = new DecimalFormat("0.000000");

	protected static final Map<String, Double> defaultLearningRate;
	protected static final Map<String, String> learningAlgMap;

	static {
		defaultLearningRate = new HashMap<String, Double>();
		defaultLearningRate.put("S", 0.1);
		defaultLearningRate.put("R", 0.1);
		defaultLearningRate.put("Q", 2.0);
		defaultLearningRate.put("B", 0.01);
		defaultLearningRate.put("M", 0.00001);

		learningAlgMap = new HashMap<String, String>();
		learningAlgMap.put("S", "Scaled Conjugate Gradient");
		learningAlgMap.put("R", "Resilient Propagation");
		learningAlgMap.put("M", "Manhattan Propagation");
		learningAlgMap.put("B", "Back Propagation");
		learningAlgMap.put("Q", "Quick Propagation");

	}

	public double getTestSetError(BasicNetwork network, MLDataSet dataSet) {
		return calculateMSE(network, dataSet);
		// return calculateMSEParallel(this.network, this.validSet);
	}

	private Double calculateMSE(BasicNetwork network, MLDataSet dataSet) {

		double mse = 0;
		long numRecords = dataSet.getRecordCount();
		for (int i = 0; i < numRecords; i++) {

			// Encog 3.1
			// MLDataPair pair = dataSet.get(i);

			// Encog 3.0
			double[] input = new double[dataSet.getInputSize()];
			double[] ideal = new double[1];
			MLDataPair pair = new BasicMLDataPair(new BasicMLData(input),
					new BasicMLData(ideal));

			dataSet.getRecord(i, pair);

			MLData result = network.compute(pair.getInput());

			double tmp = result.getData()[0] - pair.getIdeal().getData()[0];
			mse += tmp * tmp;
		}
		mse = mse / numRecords;

		return mse;
	}

	protected void splitDataSet(MLDataSet fullDataSet, Double splitRatio,
			MLDataSet trainDataSet, MLDataSet testDataSet) {

		Random random = new Random();

		for (MLDataPair pair : fullDataSet) {
			if (random.nextDouble() < splitRatio) {
				trainDataSet.add(pair);
			} else {
				testDataSet.add(pair);
			}
		}
	}

	protected MLDataSet convertDataSet(PMMLDataSet pmmlDataSet) {
		MLDataSet convertedDataSet = new BasicMLDataSet();

		List<MiningField> miningFields = pmmlDataSet.getMiningSchema()
				.getMiningFields();
		Integer numFields = miningFields.size();
		Integer numActiveFields = PMMLUtils
				.getNumActiveMiningFields(pmmlDataSet.getMiningSchema());
		Integer numTargetFields = PMMLUtils
				.getNumTargetMiningFields(pmmlDataSet.getMiningSchema());

		for (List<Object> row : pmmlDataSet.getRows()) {

			if (numFields != row.size()) {
				throw new RuntimeException(
						"MiningSchema does not match data: Number of MiningFields = "
								+ numFields + ", Number of data fields = "
								+ row.size());
			}

			double[] input = new double[numActiveFields];
			double[] ideal = new double[numTargetFields];

			int inputPtr = 0;
			int idealPtr = 0;

			for (int i = 0; i < numFields; i++) {
				if (miningFields.get(i).getUsageType()
						.equals(FieldUsageType.ACTIVE)) {
					input[inputPtr] = Double.valueOf(row.get(i).toString());
					inputPtr += 1;
				} else if (miningFields.get(i).getUsageType()
						.equals(FieldUsageType.TARGET)) {
					ideal[idealPtr] = Double.valueOf(row.get(i).toString());
					idealPtr += 1;
				}
			}

			MLDataPair pair = new BasicMLDataPair(new BasicMLData(input),
					new BasicMLData(ideal));

			convertedDataSet.add(pair);
		}

		return convertedDataSet;
	}

	protected abstract void saveEncogModel(String path, Object model);
}
