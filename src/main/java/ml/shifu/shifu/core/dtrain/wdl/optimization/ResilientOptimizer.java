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

import ml.shifu.shifu.core.dtrain.DTrainUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Resilient Optimizer(Rprop)
 * For the first iteration, it should fall back to gradient descent optimizer.
 * For the second iteration and following iteration, it update the gradient fast speed according to the logical:
 * (1) If current gradient and the before gradient are the same direction, then large the speed, usually 1.2X.
 * (2) If current gradient and the before gradient are the opposite direction, the speed rate should plus 0.5.
 * (3) Else stay the same level as gradient descent optimizer.
 * See RProp method in wiki: https://en.wikipedia.org/wiki/Rprop
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class ResilientOptimizer implements Optimizer {
    @SuppressWarnings("unused")
    private static final Logger LOG = LoggerFactory.getLogger(ResilientOptimizer.class);
    private static final double INCREASE = 1.2f;
    private static final double DECREASE = 0.5f;

    private static Map<String, Double> lastGradientMap = new HashMap<>();
    @SuppressWarnings("unused")
    private Random random = new Random(System.currentTimeMillis());
    private double learningRate;

    public ResilientOptimizer(double learningRate) {
        this.learningRate = (double) learningRate;
    }

    private double getLastGradient(String uniqueKey) {
        return lastGradientMap.get(uniqueKey);
    }

    private boolean hasGradientValue(String uniqueKey) {
        return lastGradientMap.containsKey(uniqueKey);
    }

    private void saveCurrentGradient(String uniqueKey, double gradient) {
        lastGradientMap.put(uniqueKey, gradient);
    }

    @Override
    public void update(double[] weight, double[] grad, String uniqueKey, double trainCount) {
        if(weight == null || weight.length == 0 || grad == null || grad.length != weight.length) {
            return;
        }
        for(int i = 0; i < weight.length; i++) {
            weight[i] -= updateWeight(grad[i], uniqueKey + i, trainCount);
        }
    }

    @Override
    public void update(double[] weight, Map<Integer, Double> grad, String uniqueKey, double trainCount) {
        if(weight == null || weight.length == 0 || grad == null || grad.size() == 0) {
            return;
        }

        for(Map.Entry<Integer, Double> entry: grad.entrySet()) {
            int i = entry.getKey();
            if(i < weight.length) {
                weight[i] -= updateWeight(entry.getValue(), uniqueKey + i, trainCount);
            }
        }
    }

    @Override
    public double updateWeight(double gradient, String uniqueKey, double trainCount) {
        final int change = DTrainUtils.sign(gradient * getLastGradient(uniqueKey));
        double gradientWeightUpdate = this.learningRate * gradient;

        if(!hasGradientValue(uniqueKey) || change == 0) {
            // Fall back to gradient method
            return gradientWeightUpdate;
        } else if(change > 0) {
            return gradientWeightUpdate * INCREASE;
        } else if(change < 0) {
            return gradientWeightUpdate * DECREASE;
        }
        // Save current gradient value
        saveCurrentGradient(uniqueKey, gradient);
        return gradientWeightUpdate;
    }

}
