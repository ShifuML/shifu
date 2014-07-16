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
import java.util.ArrayList;
import java.util.List;

import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.trainer.StaticFunctions.EvalMetricsCalculator;
import ml.shifu.plugin.spark.trainer.StaticFunctions.ObjectParsePoint;
import ml.shifu.plugin.spark.trainer.StaticFunctions.SumMSECalculator;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.dmg.pmml.FieldUsageType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.math.DoubleMath;

public class SparkLRTrainer extends SparkAbstractTrainer {

	private static Logger log = LoggerFactory.getLogger(SparkLRTrainer.class);
	private static final DecimalFormat df = new DecimalFormat("0.000000");
	private LogisticRegressionModel lrModel;
	private List<List<Object>> trainDataSet = new ArrayList<List<Object>>();
	private List<List<Object>> testDataSet = new ArrayList<List<Object>>();
	private int targetID;
	private int[] activeFields;

	public Object train(PMMLDataSet dataSet, Params rawParams) throws Exception {
		activeFields = SparkCommonUtil.getActiveFields(dataSet
				.getMiningSchema());
		targetID = SparkCommonUtil.getFieldIDViaUsageType(
				dataSet.getMiningSchema(), FieldUsageType.TARGET).get(0);
		SparkLRParams lrParams = parseModelParams(rawParams);
		// prepare data set
		List<List<Object>> data = dataSet.getRows();
		splitDataSet(data, lrParams.getSplitRatio(), trainDataSet, testDataSet);

		trainModel(trainDataSet, lrParams);

		// evaluate and calculate errors
		String trainerID = rawParams.get(TRAINERID).toString();
		// log.info("  Trainer-" + trainerID + "\n Train Error: "
		// + df.format(getTestSetError()));
		// calculateEvaluationScore();
		log.info("Trainer #" + trainerID + " is Finished!");
		return lrModel;
	}

	private void trainModel(List<List<Object>> trainDataSet,
			SparkLRParams lrParams) {
		JavaRDD<List<Object>> distData = SparkUtility.getSc().parallelize(
				trainDataSet);
		RDD<LabeledPoint> datas = distData
				.map(new ObjectParsePoint(targetID, activeFields)).cache()
				.rdd();
		// train model

		// for (int i = 0; i < lrParams.getIterations(); i++) {
		lrModel = LogisticRegressionWithSGD.train(datas,
				lrParams.getIterations(), lrParams.getStepSize(),
				lrParams.getSplitRatio()).clearThreshold();
		log.info(calculateTestError(testDataSet));
		// }
	}

	private SparkLRParams parseModelParams(Params rawParams) {
		ObjectMapper jsonMapper = new ObjectMapper();
		String jsonString;
		try {
			jsonString = jsonMapper.writeValueAsString(rawParams);
			return jsonMapper.readValue(jsonString, SparkLRParams.class);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private String calculateTestError(List<List<Object>> testDataSet) {
		JavaRDD<List<Object>> distData = SparkUtility.getSc().parallelize(
				testDataSet);
		JavaRDD<LabeledPoint> datas = distData.map(new ObjectParsePoint(
				targetID, activeFields));
		RDD<Tuple2<Object, Object>> matrixInput = datas
				.map(new EvalMetricsCalculator(lrModel)).cache().rdd();
		BinaryClassificationMetrics bcMetrics = new BinaryClassificationMetrics(
				matrixInput);
		StringBuilder sb = new StringBuilder();
		// EncogDirectoryPersistence.saveObject(new File(path), network);
		sb.append("AUC: " + df.format(bcMetrics.areaUnderPR()) + "\n");
		sb.append("ROC: " + df.format(bcMetrics.areaUnderROC()) + "\n");

		List<Double> results = datas.map(new SumMSECalculator(lrModel))
				.collect();
		sb.append("MSE: " + df.format(DoubleMath.mean(results)) + "\n");

		return sb.toString();
	}

	@Override
	protected void saveSparkModel(String path, Object model) {
		// TODO Auto-generated method stub

	}
}
