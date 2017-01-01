/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.pmml;

import org.dmg.pmml.Model;

/**
 * The interface that converts the Machine Learning model to a PMML model
 * 
 * @param <T>
 *            The target PMML model type
 * @param <S>
 *            The source ML model from specific Machine Learning framework such
 *            as Encog, Machout, and Spark.
 */
public interface PMMLModelBuilder<T extends Model, S> {

	T adaptMLModelToPMML(S mlModel, T partialPMMLModel);

}
