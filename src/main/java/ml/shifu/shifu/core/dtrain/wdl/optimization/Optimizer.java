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
 * @author juguo
 *
 */
public interface Optimizer {

    /**
     * Set train size
     *
     * @param trainSize
     */
    void setTrainSize(float trainSize);

    /**
     * Update gradient in @param weight
     * 
     * @param weight
     *          weight to be updated
     * @param grad
     *          the gradients
     * @param uniqueKey
     *          unique key identify the call upstream
     */
    void update(float[] weight, float[] grad, String uniqueKey);


    /**
     * Update gradient in @param weight
     *
     * @param weight
     *          weight to be updated
     * @param grad
     *          sparse representation of gradients
     * @param uniqueKey
     *          unique key identify the call upstream, usually the class name
     */
    void update(float[] weight, Map<Integer, Float> grad, String uniqueKey);

    /**
     * Update gradient in @param weight
     *
     * @param gradient
     *          gradient value
     * @param uniqueKey
     *          unique key identify the call upstream, usually the class name
     * @return
     *          weight update
     */
    float updateWeight(float gradient, String uniqueKey);

    default void batchUpdate(float[][] weights, float[][] grads, String uniqueKey) {
        if(weights == null || weights.length == 0 || grads == null || weights.length != grads.length) {
            System.out.println("Error when batch update, return");
            return;
        }
        int in = weights.length;
        for(int i = 0; i < in; i++) {
            update(weights[i], grads[i], "batchUpdateList" + uniqueKey + i);
        }
    }

    default void batchUpdate(float[][] weights, Map<Integer, float[]> grads, String uniqueKey) {
        if(weights == null || weights.length == 0 || grads == null || grads.size() == 0) {
            return;
        }
        int in = weights.length;
        for(Entry<Integer, float[]> entry: grads.entrySet()) {
            int index = entry.getKey();
            if(index < in) {
                update(weights[index], entry.getValue(), "batchUpdateMap" + uniqueKey + index);
            }
        }
    }
}
