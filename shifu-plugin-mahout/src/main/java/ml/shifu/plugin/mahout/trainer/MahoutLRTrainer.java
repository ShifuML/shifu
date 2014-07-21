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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.core.container.NNParams;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;

import org.apache.mahout.classifier.sgd.ElasticBandPrior;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.PriorFunction;
import org.apache.mahout.classifier.sgd.TPrior;
import org.apache.mahout.classifier.sgd.UniformPrior;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

public class MahoutLRTrainer extends MahoutAbstractTrainer {

	// private static final String PRIORFUNCTION = "PriorFunction";
	// private static final String DECISIONFOREST = "DecisionForest";
	// private static final String NUM_HIDDEN_NODES = "NumHiddenNodes";
	private static Logger log = LoggerFactory.getLogger(MahoutLRTrainer.class);
	private List<MahoutDataPair> fullDataSet = new ArrayList<MahoutDataPair>();
	private OnlineLogisticRegression lrModel;
	@SuppressWarnings("serial")
	private static Map<String, PriorFunction> priorFunctionMap = new HashMap<String, PriorFunction>() {
		{
			put("ElasticBandPrior", new ElasticBandPrior());
			put("L1", new L1());
			put("L2", new L2());
			put("UniformPrior", new UniformPrior());
		}
	};

	public Object train(PMMLDataSet dataSet, Params rawParams) throws Exception {
		MahoutLRParams params = parseParams(rawParams);
		String trainerID = rawParams.get("trainerID").toString();
		String pathOutput = rawParams.get("pathOutput").toString();
		File outputFolder = new File(pathOutput);
		if (!outputFolder.exists()) {
			outputFolder.mkdirs();
		}
		Integer numActiveFields = PMMLUtils.getNumActiveMiningFields(dataSet
				.getMiningSchema());
		Integer numTargetFields = PMMLUtils.getNumTargetMiningFields(dataSet
				.getMiningSchema());
		// prepare data set
		convertDataSet(dataSet, numActiveFields, numTargetFields);
		splitDataSet(params.getSplitRatio());
		// create neural network
		OnlineLogisticRegression network = createLRModel(params,
				numActiveFields);
		// train the data
		for (MahoutDataPair input : fullDataSet) {
			if (!input.isEvalData)
				network.train((int) input.getIdealData()[0],
						input.getMahoutEvalVector());
		}
		// save neural network
		String path = pathOutput;
		saveMLModel(path);
		// evaluate and calculate errors
		String extra = " <-- NN saved: " + path;
		log.info("  Trainer-" + trainerID + "\n Train Error: "
				+ df.format(getTestSetError()) + "\n" + extra);
		log.info("Trainer #" + trainerID + " is Finished!");
		return lrModel;
	}

	private Double calculateMSE(OnlineLogisticRegression network) {
		double mseError = 0;
		long numRecords = fullDataSet.size();
		for (MahoutDataPair pair : fullDataSet) {
			if (!pair.isEvalData)
				continue;
			double predict = network.classifyScalar(pair.getMahoutEvalVector());
			double idealData = pair.getIdealData()[0];
			mseError += Math.pow(idealData - predict, 2.0);
		}
		return mseError / numRecords;
	}

	private MahoutLRParams parseParams(Params rawParams) throws Exception {
		ObjectMapper jsonMapper = new ObjectMapper();
		String jsonString = jsonMapper.writeValueAsString(rawParams);
		return jsonMapper.readValue(jsonString, MahoutLRParams.class);
	}

	private OnlineLogisticRegression createLRModel(MahoutLRParams params,
			int inputSize) {
		String priorFunction = params.getPriorFunction();
		for (Map.Entry<String, PriorFunction> entry : priorFunctionMap
				.entrySet()) {
			if (priorFunction.equalsIgnoreCase(entry.getKey())) {
				lrModel = new OnlineLogisticRegression(2, inputSize,
						entry.getValue());
				return lrModel;
			}
		}
		if (priorFunction.equalsIgnoreCase("TPrior")) {
			double df = Double.parseDouble(params.gettPrior());
			lrModel = new OnlineLogisticRegression(2, inputSize, new TPrior(df));
			return lrModel;
		}
		lrModel = new OnlineLogisticRegression(2, inputSize, new L1());
		return lrModel;
	}

	private double getTestSetError() {
		return calculateMSE(this.lrModel);
		// return calculateMSEParallel(this.network, this.validSet);
	}

	private void saveMLModel(String path) throws IOException {

		// EncogDirectoryPersistence.saveObject(new File(path), network);
	}

}
