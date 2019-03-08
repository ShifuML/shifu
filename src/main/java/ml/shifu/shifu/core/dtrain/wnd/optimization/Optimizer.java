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
package ml.shifu.shifu.core.dtrain.wnd.optimization;

import java.util.Map;
import java.util.Map.Entry;

/**
 * @author juguo
 *
 */
public interface Optimizer {

    /**
     * @return the learning rate for this optimizer
     */
    double getLearningRate();

    /**
     * In some optimizer, learning rate will be updated as well.
     * 
     * @param learningRate
     *            the new learning rate
     */
    void setLearningRate(double learningRate);

    /**
     * Update gradient in @param weight
     * 
     * @param weight
     *            weight to be updated
     * @param grad
     *            the gradients
     */
    void update(float[] weight, float[] grad);

    /**
     * 
     * @param weight
     *            weight to be updated
     * @param grad
     *            sparse representation of gradients
     */
    void update(float[] weight, Map<Integer, Float> grad);

    default void batchUpdate(float[][] weights, float[][] grads) {
        if(weights == null || weights.length == 0 || grads == null || weights.length != grads.length) {
            return;
        }
        int in = weights.length;
        for(int i = 0; i < in; i++) {
            update(weights[i], grads[i]);
        }
    }

    default void batchUpdate(float[][] weights, Map<Integer, float[]> grads) {
        if(weights == null || weights.length == 0 || grads == null || grads.size() == 0) {
            return;
        }
        int in = weights.length;
        for(Entry<Integer, float[]> entry: grads.entrySet()) {
            int index = entry.getKey();
            if(index < in) {
                update(weights[index], entry.getValue());
            }
        }
    }
}
