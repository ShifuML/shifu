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

import java.util.Map;
import java.util.Map.Entry;

/**
 * @author Guo, Junshi
 *
 */
public interface Optimizer {

    /**
     * Update gradient in @param weight
     * 
     * @param weight
     *            weight to be updated
     * @param grad
     *            the gradients
     * @param uniqueKey
     *            unique key identify the call upstream
     * @param trainCount
     *            total training count which used for learning rate adjustment
     */
    void update(double[] weight, double[] grad, String uniqueKey, double trainCount);

    /**
     * Update gradient in @param weight
     *
     * @param weight
     *            weight to be updated
     * @param grad
     *            sparse representation of gradients
     * @param uniqueKey
     *            unique key identify the call upstream, usually the class name
     * @param trainCount
     *            total training count which used for learning rate adjustment
     */
    void update(double[] weight, Map<Integer, Double> grad, String uniqueKey, double trainCount);

    /**
     * Update gradient in @param weight
     *
     * @param gradient
     *            gradient value
     * @param uniqueKey
     *            unique key identify the call upstream, usually the class name
     * @param trainCount
     *            total training count which used for learning rate adjustment
     * @return
     *         weight update
     */
    double updateWeight(double gradient, String uniqueKey, double trainCount);

    default void batchUpdate(double[][] weights, double[][] grads, String uniqueKey, double trainCount) {
        if(weights == null || weights.length == 0 || grads == null || weights.length != grads.length) {
            System.out.println("Error when batch update, return");
            return;
        }
        int in = weights.length;
        for(int i = 0; i < in; i++) {
            update(weights[i], grads[i], "bl" + uniqueKey + i, trainCount);
        }
    }

    default void batchUpdate(double[][] weights, Map<Integer, double[]> grads, String uniqueKey, double trainCount) {
        if(weights == null || weights.length == 0 || grads == null || grads.size() == 0) {
            return;
        }
        int in = weights.length;
        for(Entry<Integer, double[]> entry: grads.entrySet()) {
            int index = entry.getKey();
            if(index < in) {
                update(weights[index], entry.getValue(), "bm" + uniqueKey + index, trainCount);
            }
        }
    }
}
