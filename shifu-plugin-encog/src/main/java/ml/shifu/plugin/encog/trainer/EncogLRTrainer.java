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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;

import ml.shifu.core.container.NNParams;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.util.Params;

import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

public class EncogLRTrainer extends EncogAbstractTrainer {

	public static final String ACTIVATION_FUNC = "ActivationFunc";

	public static final String PROPAGATION = "Propagation";

	private static Logger log = LoggerFactory.getLogger(EncogLRTrainer.class);

	private static final Map<String, Double> defaultLearningRate;
	private static final Map<String, String> learningAlgMap;

	private static final DecimalFormat df = new DecimalFormat("0.000000");

	private MLDataSet trainDataSet;
	private MLDataSet testDataSet;

	private Double minError;

	private BasicNetwork network;

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

	public Object train(PMMLDataSet dataSet, Params rawParams) throws Exception {

		NNParams params = parseParams(rawParams);
		MLDataSet fullDataSet = convertDataSet(dataSet);

		trainDataSet = new BasicMLDataSet();
		testDataSet = new BasicMLDataSet();

		splitDataSet(fullDataSet, params.getSplitRatio(), trainDataSet,
				testDataSet);

		log.info("    - Input Size: " + trainDataSet.getInputSize()
				+ " - Record Count: " + trainDataSet.getRecordCount());
		log.info("    - Ideal Size: " + testDataSet.getIdealSize()
				+ " - Record Count: " + testDataSet.getRecordCount());

		BasicNetwork network = createNetwork(params);
		Propagation mlTrain = getMLTrain(network, trainDataSet, params);
		mlTrain.setThreadCount(0);

		int epochs = params.getNumEpochs();
		@SuppressWarnings("unused")
		int factor = Math.max(epochs / 50, 10);

		minError = Double.MAX_VALUE;
		String trainerID = rawParams.get("trainerID").toString();
		for (int i = 0; i < epochs; i++) {
			mlTrain.iteration();
			double testError = (testDataSet.getRecordCount() > 0) ? getTestSetError()
					: mlTrain.getError();
			String extra = "";
			if (testError < minError) {
				minError = testError;
			}

			log.info("  Trainer-"
					+ trainerID
					+ "> Epoch #"
					+ (i + 1)
					+ " Train Error: "
					+ df.format(mlTrain.getError())
					+ " Test Error: "
					+ ((testDataSet.getRecordCount() > 0) ? df
							.format(testError) : "N/A") + " " + extra);

		}

		mlTrain.finishTraining();
		log.info("Trainer #" + trainerID + " is Finished!");
		// save model
		String output = rawParams.get(PATH_OUTPUT).toString();
		saveEncogModel(output, network);
		log.info("Save Encog LR model at " + output);
		return network;
	}

	private NNParams parseParams(Params rawParams) throws Exception {
		ObjectMapper jsonMapper = new ObjectMapper();
		String jsonString = jsonMapper.writeValueAsString(rawParams);
		return jsonMapper.readValue(jsonString, NNParams.class);
	}

	private BasicNetwork createNetwork(NNParams params) {
		network = new BasicNetwork();

		network.addLayer(new BasicLayer(new ActivationLinear(), true,
				trainDataSet.getInputSize()));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false,
				trainDataSet.getIdealSize()));

		network.getStructure().finalizeStructure();

		return network;
	}

	private Propagation getMLTrain(BasicNetwork network, MLDataSet trainSet,
			NNParams params) {
		// String alg = this.modelConfig.getLearningAlgorithm();
		String algorithm = params.getAlgorithm();
		if (!(defaultLearningRate.containsKey(algorithm))) {
			throw new RuntimeException("Leanring Algorithm is not valid: "
					+ algorithm);
		}

		Double rate = params.getLearningRate();
		if (rate == null) {
			rate = defaultLearningRate.get(algorithm);
		}

		log.info("    - Learning Algorithm: " + learningAlgMap.get(algorithm));
		if (algorithm.equals("Q") || algorithm.equals("B")
				|| algorithm.equals("M")) {
			log.info("    - Learning Rate: " + rate);
		}

		if (algorithm.equals("B")) {
			return new Backpropagation(network, trainSet, rate, 0);
		} else if (algorithm.equals("Q")) {
			return new QuickPropagation(network, trainSet, rate);
		} else if (algorithm.equals("M")) {
			return new ManhattanPropagation(network, trainSet, rate);
		} else if (algorithm.equals("R")) {
			return new ResilientPropagation(network, trainSet);
		} else if (algorithm.equals("S")) {
			return new ScaledConjugateGradient(network, trainSet);
		} else {
			return null;
		}
	}

	private double getTestSetError() {
		return calculateMSE(this.network, testDataSet);
		// return calculateMSEParallel(this.network, this.validSet);
	}

	private static Double calculateMSE(BasicNetwork network, MLDataSet dataSet) {

		double mse = 0;
		long numRecords = dataSet.getRecordCount();
		for (int i = 0; i < numRecords; i++) {

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

	@Override
	protected void saveEncogModel(String path, Object model) {
		try {
			new PersistBasicNetwork().save(
					new FileOutputStream(new File(path)), model);
		} catch (FileNotFoundException e) {

			e.printStackTrace();
		}

	}

}
