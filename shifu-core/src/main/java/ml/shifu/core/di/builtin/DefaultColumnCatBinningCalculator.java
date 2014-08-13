/**
 * Copyright [2012-2014] eBay Software Foundation
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

package ml.shifu.core.di.builtin;

import ml.shifu.core.container.CategoricalValueObject;
import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.di.spi.ColumnCatBinningCalculator;

import java.util.*;

public class DefaultColumnCatBinningCalculator implements ColumnCatBinningCalculator {

    public ColumnBinningResult calculate(List<CategoricalValueObject> voList) {
        ColumnBinningResult columnBinningResult = new ColumnBinningResult();

        Map<String, Integer> categoryHistNeg = new HashMap<String, Integer>();
        Map<String, Integer> categoryHistPos = new HashMap<String, Integer>();
        Map<String, Double> categoryWeightedNeg = new HashMap<String, Double>();
        Map<String, Double> categoryWeightedPos = new HashMap<String, Double>();


        List<String> binCategory = new ArrayList<String>();
        List<Integer> binCountNeg = new ArrayList<Integer>();
        List<Integer> binCountPos = new ArrayList<Integer>();
        List<Double> binPosRate = new ArrayList<Double>();
        List<Double> binWeightedNeg = new ArrayList<Double>();
        List<Double> binWeightedPos = new ArrayList<Double>();

        Set<String> categorySet = new HashSet<String>();

        int voSize = voList.size();

        for (CategoricalValueObject vo : voList) {

            String category = vo.getValue();
            categorySet.add(category);

            if (vo.getIsPositive()) {
                incMapCnt(categoryHistPos, category);
                incMapWithValue(categoryWeightedPos, category, vo.getWeight());
            } else {
                incMapCnt(categoryHistNeg, category);
                incMapWithValue(categoryWeightedNeg, category, vo.getWeight());
            }
        }

        Map<String, Double> categoryPosRateMap = new HashMap<String, Double>();

        for (String key : categorySet) {
            double cnt0 = categoryHistNeg.containsKey(key) ? categoryHistNeg.get(key) : 0;
            double cnt1 = categoryHistPos.containsKey(key) ? categoryHistPos.get(key) : 0;
            double rate;
            if (cnt0 + cnt1 == 0) {
                rate = 0;
            } else {
                rate = cnt1 / (cnt0 + cnt1);
            }
            categoryPosRateMap.put(key, rate);
        }

        // Sort map
        MapComparator cmp = new MapComparator(categoryPosRateMap);
        Map<String, Double> sortedCategoryFraudRateMap = new TreeMap<String, Double>(cmp);
        sortedCategoryFraudRateMap.putAll(categoryPosRateMap);

        for (String key : sortedCategoryFraudRateMap.keySet()) {
            Integer countNeg = categoryHistNeg.containsKey(key) ? categoryHistNeg.get(key) : 0;
            binCountNeg.add(countNeg);
            Integer countPos = categoryHistPos.containsKey(key) ? categoryHistPos.get(key) : 0;
            binCountPos.add(countPos);

            Double weightedNeg = categoryWeightedNeg.containsKey(key) ? categoryWeightedNeg.get(key) : 0.0;
            binWeightedNeg.add(weightedNeg);

            Double weightedPos = categoryWeightedPos.containsKey(key) ? categoryWeightedPos.get(key) : 0.0;
            binWeightedPos.add(weightedPos);

            // use zero, the average score is calculate in post-process

            binCategory.add(key);
            binPosRate.add(sortedCategoryFraudRateMap.get(key));
        }

        columnBinningResult.setLength(binCategory.size());
        columnBinningResult.setBinCategory(binCategory);
        columnBinningResult.setBinCountNeg(binCountNeg);
        columnBinningResult.setBinCountPos(binCountPos);
        columnBinningResult.setBinPosRate(binPosRate);
        columnBinningResult.setBinWeightedNeg(binWeightedNeg);
        columnBinningResult.setBinWeightedPos(binWeightedPos);
        /*
        for (ValueObject vo : voList) {
            String key = vo.getRaw();

            // TODO: Delete this after categorical data is correctly labeled.
            if (binCategory.indexOf(key) == -1) {
                vo.setValue(0.0);
            } else {
                // --- end deletion ---
                vo.setValue(binPosCaseRate.get(binCategory.indexOf(key)));
            }
        }     */

        return columnBinningResult;
    }

    private void incMapCnt(Map<String, Integer> map, String key) {
        int cnt = map.containsKey(key) ? map.get(key) : 0;
        map.put(key, cnt + 1);
    }

    private void incMapWithValue(Map<String, Double> map, String key, Double value) {
        double num = map.containsKey(key) ? map.get(key) : 0.0;
        map.put(key, num + value);
    }

    private class MapComparator implements Comparator<String> {
        Map<String, Double> base;

        public MapComparator(Map<String, Double> base) {
            this.base = base;
        }

        public int compare(String a, String b) {
            return base.get(a).compareTo(base.get(b));
        }
    }
}
