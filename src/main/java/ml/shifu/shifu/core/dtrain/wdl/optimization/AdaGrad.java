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
public class AdaGrad implements Optimizer {

    private double learningRate;

    public AdaGrad(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(float[] weight, float[] grad, String uniqueKey) {
        if(weight == null || grad == null || grad.length == 0 || weight.length != grad.length) {
            return;
        }
        int len = weight.length;

        double sumG2 = 0;
        for(int i = 0; i < len; i++) {
            sumG2 += grad[i] * grad[i];
        }
        double sumG2Sqrt = Math.sqrt(sumG2) + 0.000001;

        for(int i = 0; i < len; i++) {
            double delta = learningRate * grad[i] / sumG2Sqrt;
            weight[i] -= delta;
        }
    }

    @Override
    public void update(float[] weight, Map<Integer, Float> grad, String uniqueKey) {
        if(weight == null || weight.length == 0 || grad == null || grad.size() == 0) {
            return;
        }

        double sumG2 = 0;
        for(Entry<Integer, Float> entry: grad.entrySet()) {
            sumG2 += entry.getValue() * entry.getValue();
        }
        double sumG2Sqrt = Math.sqrt(sumG2) + 0.000001;

        int len = weight.length;
        for(Entry<Integer, Float> entry: grad.entrySet()) {
            int index = entry.getKey();
            double delta = learningRate * entry.getValue() / sumG2Sqrt;
            if(index < len) {
                weight[index] -= delta;
            }
        }
    }

    @Override
    public float updateWeight(float gradient, String uniqueKey) {
        return Double.valueOf(this.learningRate * gradient).floatValue();
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

}
