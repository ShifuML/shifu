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

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class SparkLRParams {
	private int iterations;
	private Double stepSize;
	private double splitRatio;

	public double getSplitRatio() {
		return splitRatio;
	}

	public void setSplitRatio(double splitRatio) {
		this.splitRatio = splitRatio;
	}

	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	public Double getStepSize() {
		return stepSize;
	}

	public void setStepSize(Double stepSize) {
		this.stepSize = stepSize;
	}

}
