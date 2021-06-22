/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.udf.stats;

import ml.shifu.shifu.util.BinUtils;

import java.util.*;

/**
 * Created by Mark on 5/27/2016.
 */
public class NumericCounter extends Counter {

    private List<Double> binBoundary;

    @SuppressWarnings("unused")
    private String name;

    public NumericCounter(List<String> missingInvalidValues, String name, List<Double> binBoundary) {
        super(binBoundary.size(), new HashSet<>(missingInvalidValues));
        this.name = name;
        this.binBoundary = binBoundary;
    }

    @Override
    public void addData(Boolean isPositive, String val) {
        long[] counter = (isPositive ? positiveCounter : negativeCounter);

        if(val == null || missingValSet.contains(val)) {
            counter[binLen] += 1;
        } else {
            try {
                Double dVal = Double.parseDouble(val);
                int index = BinUtils.getBinIndex(binBoundary, dVal);
                counter[index] += 1;
                unitSum += dVal;
            } catch (Exception e) {
                // logger.warn("Unable to count this column {} with {}, using default value", name, val);
                counter[binLen] = counter[binLen] + 1;
            }
        }
    }

    // The instances that value are missing, shouldn't be used to calculate mean
    @Override
    public double getUnitMean() {
        long total = getTotalInstCnt() - getTotalMissingCnt();

        double unitMean;
        if(total <= 0) {
            // no instance or all missing
            unitMean = Double.NaN;
        } else {
            unitMean = this.unitSum / total;
        }

        return unitMean;
    }

}
