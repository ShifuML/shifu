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

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class MahoutDataPair {
	double[] inputData;
	double[] outputData;
	boolean isEvalData = false;

	public MahoutDataPair(double[] data, double[] ideal) {
		inputData = data;
		outputData = ideal;
	}

	public MahoutDataPair(double[] data, double[] ideal, boolean isEvalData) {
		inputData = data;
		outputData = ideal;
		this.isEvalData = isEvalData;
	}

	public Vector getMahoutInputVector() {
		int inputLen = inputData.length;
		double[] inputList = new double[inputLen + outputData.length];
		for (int i = 0; i < inputLen; i++)
			inputList[i] = inputData[i];
		for (int i = 0; i < outputData.length; i++)
			inputList[i + inputLen] = outputData[i];
		return new DenseVector(inputList);
	}

	public Vector getMahoutEvalVector() {

		return new DenseVector(inputData);
	}

	public double[] getIdealData() {
		return outputData;
	}

	public boolean isEvalData() {
		return isEvalData;
	}

	public void setEvalData(boolean isEvalData) {
		this.isEvalData = isEvalData;
	}

}
