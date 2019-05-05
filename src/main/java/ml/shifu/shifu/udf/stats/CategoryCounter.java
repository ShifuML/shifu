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

    private List<String> categories;
    private List<Double> binPosRate;
    private Map<String, Integer> categoryValIndex;

    public CategoryCounter(List<String> missingInvalidValues, List<String> categories, List<Double> binPosRate) {
        super(categories.size(), new HashSet<>(missingInvalidValues));

        this.categories = categories;
        this.binPosRate = binPosRate;

        this.categoryValIndex = new HashMap<String, Integer>();
        for(int i = 0; i < categories.size(); i++) {
            categoryValIndex.put(categories.get(i), i);
        }
    }

    @Override
    public void addData(Boolean isPositive, String val) {
        if(isPositive == null) {
            isPositive = true;
        }

        long[] counter = (isPositive ? this.positiveCounter : this.negativeCounter);

        int pos = binLen;
        if(val != null && !this.missingValSet.contains(val)) {
            Integer cidx = this.categoryValIndex.get(val);
            if (cidx != null) {
                pos = cidx;
            }
        }

        counter[pos] = counter[pos] + 1;
        this.unitSum += this.binPosRate.get(pos);
    }
}
