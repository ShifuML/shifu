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


import ml.shifu.core.container.ColumnNumStatsResult;
import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.core.di.spi.ColumnNumStatsCalculator;
import ml.shifu.core.util.QuickSort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class DefaultColumnNumStatsCalculator implements ColumnNumStatsCalculator {

    private static Logger log = LoggerFactory.getLogger(DefaultColumnNumStatsCalculator.class);


    private Double sum = 0.0;
    private Double squaredSum = 0.0;
    private Double min = Double.MAX_VALUE;
    private Double max = -Double.MAX_VALUE;
    private Double mean = Double.NaN;
    private Double stdDev = Double.NaN;
    private Double median = Double.NaN;
    private Double EPS = 1e-6;

    public ColumnNumStatsResult calculate(List<NumericalValueObject> nvoList) {

        if (nvoList.size() == 0) {
            throw new IllegalArgumentException("Empty List");
        }

        QuickSort.sort(nvoList, new NumericalValueObject.NumericalValueObjectComparator());

        int validSize = 0;

        for (NumericalValueObject nvo : nvoList) {
            Double value = nvo.getValue();

            if (value.isInfinite() || value.isNaN()) {
                continue;
            }

            validSize += 1;
            max = Math.max(max, value);
            min = Math.min(min, value);

            sum += value;
            squaredSum += value * value;
        }

        // mean and stdDev defaults to NaN
        if (sum.isInfinite() || squaredSum.isInfinite()) {
            throw new RuntimeException("Sum or SquaredSum exceeds limit of Double");
        }

        //it's ok while the voList is sorted;
        median = nvoList.get(nvoList.size() / 2).getValue();

        mean = sum / validSize;
        stdDev = Math.sqrt((squaredSum - (sum * sum) / validSize + EPS)
                / (validSize - 1));


        ColumnNumStatsResult stats = new ColumnNumStatsResult();
        stats.setMax(max);
        stats.setMin(min);
        stats.setMean(mean);
        stats.setMedian(median);
        stats.setStdDev(stdDev);

        return stats;
    }
}
