/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wdl.optimization;

import ml.shifu.shifu.core.dtrain.RegulationLevel;

/**
 * Support different kinds of optimizations like RProp, QuickProp, BackProp ...
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public interface PropOptimizer<SELF extends PropOptimizer<?>> {
    /**
     * Init weight optimizer according to learning rate, algorithm, reg and rl
     * @param learningRate
     *                  the learning rate
     * @param algorithm
     *                  the algorithm
     * @param reg
     *                  the regulation value
     * @param rl
     *                  the regulation level
     */
    void initOptimizer(double learningRate, String algorithm, double reg, RegulationLevel rl);

    /**
     * Optimize weight
     *
     * @param numTrainSize
     *                  total training size
     */

    /**
     * Optimize Weight
     * @param numTrainSize
     *                  total training size
     * @param iteration
     *                  current iteration
     * @param model
     *                  current model
     */
    void optimizeWeight(double numTrainSize, int iteration, SELF model);
}
