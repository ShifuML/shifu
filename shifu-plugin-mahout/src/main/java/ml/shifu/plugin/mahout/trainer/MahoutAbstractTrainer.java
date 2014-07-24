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

package ml.shifu.plugin.mahout.trainer;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.spi.Trainer;

import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;

public abstract class MahoutAbstractTrainer implements Trainer {
	protected List<MahoutDataPair> fullDataSet = new ArrayList<MahoutDataPair>();
	// protected NeuralNetwork network;
	protected static final DecimalFormat df = new DecimalFormat("0.000000");

	protected void convertDataSet(PMMLDataSet pmmlDataSet, int numActiveFields,
			int numTargetFields) {
		List<MiningField> miningFields = pmmlDataSet.getMiningSchema()
				.getMiningFields();
		Integer numFields = miningFields.size();
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
			fullDataSet.add(new MahoutDataPair(input, ideal));
		}
	}

	/**
	 * Split the train dataset to train dataset and test data set based on the
	 * split ratio
	 * 
	 * @param fullDataSet
	 * @param splitRatio
	 * @param trainDataSet
	 * @param testDataSet
	 */
	protected void splitDataSet(Double splitRatio) {
		Random random = new Random();
		for (MahoutDataPair pair : fullDataSet) {
			if (random.nextDouble() <= splitRatio) {
				pair.setEvalData(true);
			}
		}
	}

}
