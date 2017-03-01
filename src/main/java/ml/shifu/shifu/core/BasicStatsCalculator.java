/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.ValueObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Calculator, it helps to calculate the sum, max, min and standard deviation for value object list
 */
public class BasicStatsCalculator {

    /**
     * logger
     */
    private static Logger log = LoggerFactory.getLogger(BasicStatsCalculator.class);

    private List<ValueObject> voList;

    private Double sum;
    private Double squaredSum;
    private Double min = Double.MAX_VALUE;
    private Double max = -Double.MAX_VALUE;
    private Double mean = Double.NaN;
    private Double stdDev = Double.NaN;
    private Double median = Double.NaN;

    private Double threshold = 1e6;
    private Double EPS = 1e-6;

    public BasicStatsCalculator(List<ValueObject> voList, Double threshold) {
        this.voList = voList;
        this.threshold = threshold;
        calculateStats();
    }

    private void calculateStats() {
        sum = 0.0;
        squaredSum = 0.0;

        if (voList.size() == 0) {
            return;
        }

        int validSize = 0;

        for (ValueObject vo : voList) {
            Double value = vo.getValue();

            if (value.isInfinite() || value.isNaN() || Math.abs(value) > threshold) {
                log.warn("Invalid value - " + value);
                continue;
            }

            validSize++;

            max = Math.max(max, value);
            min = Math.min(min, value);

            sum += value;
            squaredSum += value * value;
        }

        // mean and stdDev defaults to NaN
        if (validSize <= 1 || sum.isInfinite() || squaredSum.isInfinite()) {
            return;
        }

        //it's ok while the voList is sorted;
        setMedian(voList.get(voList.size() / 2).getValue());

        mean = sum / validSize;
        stdDev = Math.sqrt((squaredSum - (sum * sum) / validSize + EPS)
                / (validSize - 1));
    }

    public Double getMin() {
        return min;
    }

    public Double getMax() {
        return max;
    }

    public Double getMean() {
        return mean;
    }

    public Double getStdDev() {
        return stdDev;
    }

    public Double getMedian() {
        return median;
    }

    public void setMedian(Double median) {
        this.median = median;
    }

}
