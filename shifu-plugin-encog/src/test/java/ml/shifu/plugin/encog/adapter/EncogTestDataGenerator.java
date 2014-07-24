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

package ml.shifu.plugin.encog.adapter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.PMML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.jpmml.evaluator.ModelEvaluationContext;
import org.jpmml.evaluator.NormalizationUtil;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

/**
 * This class loads the test data and maps these data as the input of the
 * machine learning models.
 */
public class EncogTestDataGenerator {
	private PMML pmml;
	private String[] headers;
	private String path;
	/**
	 * for PMML evaluator
	 */
	private List<Map<FieldName, String>> table = Lists.newArrayList();
	/**
	 * for encog framework
	 */
	private List<double[]> dataSet = new ArrayList<double[]>();
	private BufferedReader reader;

	public EncogTestDataGenerator(String dataPath, PMML pmml) {
		this.path = dataPath;
		this.pmml = pmml;
		headers = PMMLAdapterCommonUtil.getDataDicHeaders(pmml);
		parseCSV();

	}

	/**
	 * Return a list of fieldName-value map that can be used as the input of
	 * PMML evaluator.
	 * 
	 * @returna The list of fieldName-value map that is used as the input of
	 *          PMML evaluator.
	 */
	public List<Map<FieldName, String>> getEvaluatorInput() {
		return table;
	}

	/**
	 * <p>
	 * Construct the Encog MLDataSet from CSV files, as the input training set
	 * for the Encog framework.
	 * <p>
	 * Notice that, the input of MLDataSet includes a bias, which is always set
	 * to 1.
	 * 
	 * @return The MLDataSet that can be used as the input training data set for
	 *         Encog framework.
	 */
	public MLDataSet getEncogMLDataSet() {
		MLDataSet encogSet = new BasicMLDataSet();
		for (double[] fields : dataSet) {
			int len = fields.length;
			double[] itemList = new double[len];
			for (int i = 0; i < len - 1; i++)
				itemList[i] = fields[i + 1];
			itemList[len - 1] = 1;
			encogSet.add(new BasicMLData(itemList));
		}
		return encogSet;
	}

	private void parseCSV() {
		try {
			reader = new BufferedReader(new FileReader(path));
			String bodyLine;
			while ((bodyLine = reader.readLine()) != null) {
				Map<FieldName, String> row = Maps.newLinkedHashMap();
				String[] bodyCells = bodyLine.split(",");

				if (bodyCells.length != headers.length) {
					throw new RuntimeException();
				}

				for (int i = 0; i < bodyCells.length; i++) {
					row.put(new FieldName(headers[i]), bodyCells[i]);
				}
				table.add(row);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * <p>
	 * Construct the Encog MLDataSet from CSV files, as the input training set
	 * for the Encog framework.
	 * <p>
	 * Notice that, the input of MLDataSet includes a bias, which is always set
	 * to 1.
	 * 
	 * @return The MLDataSet that can be used as the input training data set for
	 *         Encog framework.
	 */
	private MLData getEncogMLDataSet(List<Double> data) {
		int len = data.size();
		double[] itemList = new double[len + 1];
		for (int i = 0; i < len; i++)
			itemList[i] = data.get(i);
		itemList[len] = 1;
		return new BasicMLData(itemList);

	}

	/**
	 * normalize the input data based on the model evaluation context and
	 * returns the normalized data
	 * 
	 * @param context
	 *            model evaluation context
	 * @return the normalized data
	 */
	public MLData normalizeData(ModelEvaluationContext context) {
		Model model = pmml.getModels().get(0);
		List<DerivedField> derivedFields = model.getLocalTransformations()
				.getDerivedFields();

		List<Double> transformed = new ArrayList<Double>();
		for (DerivedField df : derivedFields) {
			if (df.getExpression() instanceof NormContinuous) {
				NormContinuous norm = (NormContinuous) df.getExpression();
				transformed.add(Double.parseDouble(NormalizationUtil
						.normalize(norm, context.getField(norm.getField()))
						.getValue().toString()));
			}
		}
		return getEncogMLDataSet(transformed);
	}
}
