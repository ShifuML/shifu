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
package ml.shifu.shifu.core.dtrain.nn.update;

/**
 * Interface Update to support different implementation of weight.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public interface Update {
    /**
     * Get learning rate
     * @return
     *          the learning rate
     */
    double getLearningRate();

    /**
     * Get weight num
     * @return
     *      the number of weights
     */
    int getNumWeight();

    /**
     * Get Adam beta1
     * @return
     *       the adam beta1
     */
    double getAdamBeta1();

    /**
     * Get Adam beta2
     * @return
     *       the adam beta2
     */
    double getAdamBeta2();

    /**
     * Get momentum
     * @return
     *       the momentum
     */
    double getMomentum();

    /**
     * Get learning decay
     * @return
     *      the learning decay
     */
    double getLearningDecay();
    
    /**
     * Get number of train size (not weighted)
     * @return
     *      the learning decay
     */
    double getNumTrainSize() ;

}
