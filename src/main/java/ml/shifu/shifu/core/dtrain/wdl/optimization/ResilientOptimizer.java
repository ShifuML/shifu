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
import static org.encog.neural.flat.train.prop.RPROPConst.DEFAULT_MAX_STEP;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Resilient Optimizer.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class ResilientOptimizer implements Optimizer{
    private static final Logger LOG = LoggerFactory.getLogger(ResilientOptimizer.class);

    private Map<String, Float> lastOptimizerMap = new HashMap<>();
    private float reg;
    private float trainSize;

    public ResilientOptimizer(float reg) {
        this.reg = reg;
    }

    private float getLastValue(String uniqueKey) {
        if(lastOptimizerMap.containsKey(uniqueKey)) {
            return lastOptimizerMap.get(uniqueKey);
        }
        LOG.error("Can not find " + uniqueKey + " in lastOptimizerMap");
        return 0;
    }

    private String getLastGradientKey(String uniqueKey) {
        return uniqueKey + "g";
    }

    private String getLastDeltaKey(String uniqueKey) {
        return uniqueKey + "d";
    }

    private String getUpdateValueKey(String uniqueKey) {
        return uniqueKey + "u";
    }

    @Override
    public void update(float[] weight, float[] grad, String uniqueKey) {
        if(weight == null || weight.length == 0 || grad == null || grad.length != weight.length) {
            return;
        }
        int len = weight.length;
        for(int i = 0; i < len; i++) {
            weight[i] += (updateWeight(grad[i], uniqueKey + i) - this.reg * weight[i] / this.trainSize);
        }
    }

    @Override
    public void update(float[] weight, Map<Integer, Float> grad, String uniqueKey) {
        if(weight == null || weight.length == 0 || grad == null || grad.size() == 0) {
            return;
        }

        for(Map.Entry<Integer, Float> entry: grad.entrySet()) {
            int i = entry.getKey();
            if(i < weight.length) {
                weight[i] += (updateWeight(entry.getValue(), uniqueKey + i) - this.reg * weight[i] / this.trainSize);
            }
        }
    }

    @Override
    public float updateWeight(float gradient, String uniqueKey) {
        String updateValueKey = getUpdateValueKey(uniqueKey);
        String lastDeltaKey = getLastDeltaKey(uniqueKey);
        String lastGradientKey = getLastGradientKey(uniqueKey);

        final int change = DTrainUtils.sign(gradient * getLastValue(lastGradientKey));
        float weightChange = 0;

        // if the gradient has retained its sign, then we increase the delta so that it will converge faster
        if(change > 0) {
            float delta = Double.valueOf(getLastValue(updateValueKey) * DTrainUtils.POSITIVE_ETA).floatValue();
            delta = Double.valueOf(Math.min(delta, DEFAULT_MAX_STEP)).floatValue();
            weightChange = Double.valueOf(DTrainUtils.sign(gradient) * delta).floatValue();
            lastOptimizerMap.put(updateValueKey, delta);
            lastOptimizerMap.put(lastGradientKey, gradient);
        } else if(change < 0) {
            // if change<0, then the sign has changed, and the last delta was too big
            float delta = Double.valueOf(getLastValue(updateValueKey) * DTrainUtils.NEGATIVE_ETA).floatValue();
            delta = Double.valueOf(Math.max(delta, DTrainUtils.DELTA_MIN)).floatValue();
            lastOptimizerMap.put(updateValueKey, delta);
            weightChange = -lastOptimizerMap.get(lastDeltaKey);
            // set the previous gradient to zero so that there will be no adjustment the next iteration
            lastOptimizerMap.put(lastGradientKey, 0f);
        } else {
            // if change==0 then there is no change to the delta
            final double delta = lastOptimizerMap.get(updateValueKey);
            weightChange = Double.valueOf(DTrainUtils.sign(gradient) * delta).floatValue();
            lastOptimizerMap.put(lastGradientKey, gradient);
        }

        lastOptimizerMap.put(lastDeltaKey, weightChange);
        // apply the weight change, if any
        return weightChange;
    }

    /**
     * @return the reg
     */
    public float getReg() {
        return reg;
    }

    /**
     * @param reg
     *      the reg to set
     */
    public void setReg(float reg) {
        this.reg = reg;
    }

    /**
     * @return the trainSize
     */
    public float getTrainSize() {
        return trainSize;
    }

    /**
     * @param trainSize
     *          the trainSize to set
     */
    @Override
    public void setTrainSize(float trainSize) {
        this.trainSize = trainSize;
    }
}
