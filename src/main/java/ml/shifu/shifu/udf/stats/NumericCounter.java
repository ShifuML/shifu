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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ml.shifu.shifu.util.CommonUtils;

/**
 * Created by Mark on 5/27/2016.
 */
public class NumericCounter extends Counter {

    private Set<String> missingValSet = new HashSet<String>();
    private List<Double> binBoundary;
    @SuppressWarnings("unused")
    private String name;

    private Long[] counter;
    private double unitSum = 0.0;

    public NumericCounter(List<String> missingInvalidValues, String name, List<Double> binBoundary) {
        this.missingValSet.addAll(missingInvalidValues);
        this.name = name;
        this.binBoundary = binBoundary;
        this.counter = new Long[binBoundary.size() + 1];
        Arrays.fill(counter, 0L);
    }

    @Override
    public void addData(String val) {
        if ( val == null || missingValSet.contains(val) ) {
            counter[binBoundary.size()] = counter[binBoundary.size()] + 1;
        } else {
            try {
                Double dVal = Double.parseDouble(val.toString());
                int index = CommonUtils.getBinIndex(binBoundary, dVal);
                counter[index] = counter[index] + 1;
                unitSum += dVal;
            } catch (Exception e) {
                // logger.warn("Unable to count this column {} with {}, using default value", name, val);
                counter[binBoundary.size()] = counter[binBoundary.size()] + 1;
            }
        }
    }

    @Override
    public List<Long> getCounter() {
        return Arrays.asList(counter);
    }

    @Override
    public double getUnitMean() {
        long total = getTotalInstCnt();

        double unitMean;
        if ( total == 0 || total == counter[binBoundary.size()] ){
            // no instance or all missing
            unitMean = Double.NaN;
        } else {
            unitMean = this.unitSum / total;
        }

        return unitMean;
    }

    @Override
    public double getMissingRate() {
        long total = getTotalInstCnt();
        double missingInstCnt = counter[binBoundary.size()];
        return ((total != 0) ? missingInstCnt/total : 0.0);
    }

    @Override
    public long getTotalInstCnt() {
        long total = 0;
        for ( Long val: counter ) {
            total += val;
        }
        return total;
    }
}
