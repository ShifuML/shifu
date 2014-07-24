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

package ml.shifu.core.plugin.pmml;

/**
 * The abstract class that converts the Machine Learing model to a PMML model
 * 
 * @param <T>
 *            The target Machine Learning model from specific Machine Learning
 *            framework such as Encog, Machout, and Spark.
 * @param <S>
 *            The source PMML model
 */
public interface GenericMLModelBuilder<T, S> {

	/**
	 * The function creates a specific Machine Learning model from PMML model.
	 * 
	 * @param pmmlModel
	 *            The model from ML frameworks
	 * @return The Machine Learning model converted from PMMl model
	 */
	T createMLModelFromPMML(S pmmlModel);

}
