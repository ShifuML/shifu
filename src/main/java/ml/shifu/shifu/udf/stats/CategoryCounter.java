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

import java.util.*;

/**
 * counter for the categorical val
 */
public class CategoryCounter extends Counter {

    private List<Double> binPosRate;
    private List<String> categories;

    private Set<String> missingValSet = new HashSet<String>();
    private Map<String, Integer> categoryValIndex = new HashMap<String, Integer>();
    private Map<String, Long> categoryMap = new HashMap<String, Long>();
    private long missCounter;
    private double unitSum = 0.0;

    public CategoryCounter(List<String> missingInvalidValues, List<String> categories, List<Double> binPosRate) {
        this.missingValSet.addAll(missingInvalidValues);

        this.categories = categories;
        this.binPosRate = binPosRate;

        for(int i = 0; i < categories.size(); i++) {
            categoryMap.put(categories.get(i), 0L);
            categoryValIndex.put(categories.get(i), i);
        }

        this.missCounter = 0;
    }

    @Override
    public void addData(String val) {
        if(val == null || this.missingValSet.contains(val)) {
            missCounter++;
        } else {
            String sVal = val.toString();
            if(categoryMap.containsKey(sVal)) {
                categoryMap.put(sVal, categoryMap.get(sVal) + 1);
                int index = categoryValIndex.get(sVal);
                this.unitSum += this.binPosRate.get(index);
            } else {
                missCounter++;
            }
        }
    }

    @Override
    public List<Long> getCounter() {
        List<Long> counters = new ArrayList<Long>();

        for(int i = 0; i < categories.size(); i++) {
            counters.add(categoryMap.get(categories.get(i)));
        }

        counters.add(missCounter);

        return counters;
    }

    @Override
    public double getUnitMean() {
        long total = getTotalInstCnt();

        double unitMean;
        if(total == 0 || total == missCounter) {
            unitMean = Double.NaN;
        } else {
            unitMean = this.unitSum / total;
        }

        return unitMean;
    }

    @Override
    public double getMissingRate() {
        long total = getTotalInstCnt();
        double missingInstCnt = missCounter;
        return ((total != 0) ? missingInstCnt / total : 0.0);
    }

    @Override
    public long getTotalInstCnt() {
        long total = 0;
        for(Long val: categoryMap.values()) {
            total += val;
        }
        return total + missCounter;
    }
}
