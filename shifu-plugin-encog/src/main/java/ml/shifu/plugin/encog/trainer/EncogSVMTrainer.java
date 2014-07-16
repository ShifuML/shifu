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
import java.util.HashMap;
import java.util.Map;

import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.container.SVMParams;
import ml.shifu.core.util.Params;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.svm.KernelType;
import org.encog.ml.svm.PersistSVM;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.SVMType;
import org.encog.ml.svm.training.SVMTrain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Implementation of AbstractTrainer for support vector machine classification
 */
public class EncogSVMTrainer extends EncogAbstractTrainer {

	public static final String SVM_KERNEL = "Kernel";
	public static final String SVM_GAMMA = "Gamma";
	public static final String SVM_CONST = "Const";
	public static final String SVM_SPLITRATIO = "splitRatio";
	private static final String TRAINERID = "trainerID";
	private SVM svm;
	private static Map<String, KernelType> kernel = new HashMap<String, KernelType>();
	private static Map<String, SVMType> type = new HashMap<String, SVMType>();
	private MLDataSet trainDataSet;
	private MLDataSet testDataSet;
	private String trainerID = "";
	protected Logger log = LoggerFactory.getLogger(EncogSVMTrainer.class);

	static {
		kernel.put("Leaner kernel".toLowerCase(), KernelType.Linear);
		kernel.put("linear".toLowerCase(), KernelType.Linear);
		kernel.put("Poly kernel".toLowerCase(), KernelType.Poly);
		kernel.put("poly", KernelType.Poly);
		kernel.put("Sigmoid kernel".toLowerCase(), KernelType.Sigmoid);
		kernel.put("Sigmoid".toLowerCase(), KernelType.Sigmoid);
		kernel.put("RadialBasisFunction".toLowerCase(),
				KernelType.RadialBasisFunction);
		kernel.put("RBF".toLowerCase(), KernelType.RadialBasisFunction);

		type.put("classification", SVMType.SupportVectorClassification);
		type.put("regresssion", SVMType.EpsilonSupportVectorRegression);
	}

	@Override
	public Object train(PMMLDataSet dataSet, Params rawParams) throws Exception {
		MLDataSet fullDataSet = convertDataSet(dataSet);
		SVMParams svmParams = parseParams(rawParams);
		trainerID = rawParams.get(TRAINERID).toString();
		trainDataSet = new BasicMLDataSet();
		testDataSet = new BasicMLDataSet();

		splitDataSet(fullDataSet, svmParams.getSplitRatio(), trainDataSet,
				testDataSet);

		encogTrain(svmParams);
		// save model
		String output = rawParams.get(PATH_OUTPUT).toString();
		saveEncogModel(output, svm);
		log.info("Save Encog SVM model at " + output);
		return svm;
	}

	private SVMParams parseParams(Params rawParams) throws Exception {
		ObjectMapper jsonMapper = new ObjectMapper();
		String jsonString = jsonMapper.writeValueAsString(rawParams);
		return jsonMapper.readValue(jsonString, SVMParams.class);
	}

	/**
	 * Setup SVM
	 */
	private void buildSVM(SVMParams params) {
		svm = new SVM(trainDataSet.getInputSize(),
				SVMType.SupportVectorClassification, kernel.get(params
						.getKernel()));
	}

	/**
	 * using Encog's SVM trainer
	 */
	private void encogTrain(SVMParams params) {
		buildSVM(params);
		log.info("    - Input Size: " + trainDataSet.getInputSize()
				+ " - Record Count: " + trainDataSet.getRecordCount());
		log.info("    - Ideal Size: " + testDataSet.getIdealSize()
				+ " - Record Count: " + testDataSet.getRecordCount());
		SVMTrain trainer = new SVMTrain(svm, trainDataSet);
		trainer.setC(params.getConstant());
		trainer.setGamma(params.getGamma());

		SVMRunner runner = new SVMRunner(trainer);
		Thread thread = new Thread(runner);
		thread.start();

		long second = 1000;

		while (!runner.isFinish()) {
			try {
				Thread.sleep(second);
				log.info("Trainer #" + this.trainerID + " is running");
			} catch (InterruptedException e) {
				throw new RuntimeException("Within system interrupted");
			}
		}

		log.info("Trainer #" + this.trainerID + " finish training");

		trainer = runner.trainer;

		log.info("Train #" + this.trainerID + " Error: "
				+ df.format(trainer.getError()) + " Validation Error:"
				+ df.format(getValidSetError()));

	}

	public double getValidSetError() {
		return svm.calculateError(this.testDataSet);
	}

	public SVM getSVM() {
		return svm;
	}

	/**
	 * SVMtrainer worker
	 */
	private class SVMRunner implements Runnable {

		private SVMTrain trainer;
		private boolean isFinish;

		public SVMRunner(SVMTrain trainer) {
			this.trainer = trainer;
			this.isFinish = false;
		}

		@Override
		public void run() {

			trainer.setFold(1);

			trainer.iteration();

			isFinish = true;
		}

		public boolean isFinish() {
			return isFinish;
		}
	}

	@Override
	protected void saveEncogModel(String path, Object model) {
		try {
			new PersistSVM().save(new FileOutputStream(new File(path)), model);
		} catch (FileNotFoundException e) {

			e.printStackTrace();
		}

	}
}
