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

import java.util.List;
import java.util.regex.Pattern;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class StaticFunctions {

	@SuppressWarnings("serial")
	public static class ParsePoint implements Function<String, LabeledPoint> {
		private Pattern COMMA;
		private int target;
		private int[] activeField;

		public ParsePoint(int targetID, int[] activeField, String splitter) {
			target = targetID;
			this.activeField = activeField;
			COMMA = Pattern.compile("\\" + splitter);
		}

		@Override
		public LabeledPoint call(String line) {
			String[] parts = COMMA.split(line);
			double y = Double.parseDouble(parts[target]);
			int len = activeField.length;
			double[] x = new double[len];
			for (int i = 0; i < len; i++) {
				x[i] = Double.parseDouble(parts[activeField[i]]);
			}
			return new LabeledPoint(y, Vectors.dense(x));
		}
	}

	@SuppressWarnings("serial")
	public static class ObjectParsePoint implements
			Function<List<Object>, LabeledPoint> {
		private int target;
		private int[] activeField;

		public ObjectParsePoint(int targetID, int[] activeField) {
			target = targetID;
			this.activeField = activeField;
		}

		@Override
		public LabeledPoint call(List<Object> line) {
			double y = Double.parseDouble(line.get(target).toString());
			int len = activeField.length;
			double[] x = new double[len];
			for (int i = 0; i < len; i++) {
				x[i] = Double.parseDouble(line.get(activeField[i]).toString());
			}
			return new LabeledPoint(y, Vectors.dense(x));
		}
	}

//	@SuppressWarnings("serial")
//	public static class ParseVector implements Function<String, Vector> {
//		private int targetID;
//		private int[] activeField;
//		private Pattern COMMA;
//
//		public ParseVector(int targetID, int[] activeField, String splitter) {
//			this.activeField = activeField;
//			this.targetID = targetID;
//			COMMA = Pattern.compile("\\" + splitter);
//		}
//
//		@Override
//		public Vector call(String line) {
//			String[] parts = COMMA.split(line);
//			int len = activeField.length;
//			double[] x = new double[len];
//			for (int i = 0; i < len; i++) {
//				x[i] = Double.parseDouble(parts[activeField[i]]);
//			}
//			return Vectors.dense(x);
//		}
//	}

	@SuppressWarnings("serial")
	public static class EvalMetricsCalculator implements
			Function<LabeledPoint, Tuple2<Object, Object>> {

		private LogisticRegressionModel lrModel;

		public EvalMetricsCalculator(LogisticRegressionModel lrModel) {
			this.lrModel = lrModel;
		}

		@Override
		public Tuple2<Object, Object> call(LabeledPoint line) throws Exception {
			return new Tuple2<Object, Object>(lrModel.predict(line.features()),
					line.label());
		}
	}

	@SuppressWarnings("serial")
	public static class SumMSECalculator implements
			Function<LabeledPoint, Double> {

		private LogisticRegressionModel lrModel;

		public SumMSECalculator(LogisticRegressionModel lrModel) {
			this.lrModel = lrModel;
		}

		@Override
		public Double call(LabeledPoint line) throws Exception {
			return Math.pow(lrModel.predict(line.features()) - line.label(), 2);
		}
	}
}
