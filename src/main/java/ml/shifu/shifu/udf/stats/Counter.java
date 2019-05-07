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

import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;

import java.util.*;

/**
 * Created by Mark on 5/27/2016.
 */
public abstract class Counter {

    protected int binLen;
    protected Set<String> missingValSet;
    protected long[] positiveCounter;
    protected long[] negativeCounter;
    protected double unitSum;

    public Counter(int binLen, Set<String> missingValSet) {
        this.binLen = binLen;
        this.missingValSet = missingValSet;
        this.positiveCounter = new long[binLen + 1];
        Arrays.fill(positiveCounter, 0L);
        this.negativeCounter = new long[binLen + 1];
        Arrays.fill(negativeCounter, 0L);
        this.unitSum = 0.0d;
    }

    public abstract void addData(Boolean tag, String val);

    public List<Long> getCounter() {
        List<Long> totalCounter = new ArrayList<>();
        for(int i = 0; i < binLen + 1; i++) {
            totalCounter.add(this.positiveCounter[i] + this.negativeCounter[i]);
        }
        return totalCounter;
    }

    public long[] getPositiveCounter() {
        return this.positiveCounter;
    }

    public long[] getNegativeCounter() {
        return this.negativeCounter;
    }

    public double getUnitMean() {
        long total = getTotalInstCnt();

        double unitMean;
        if(total == 0 || total == getTotalMissingCnt()) {
            // no instance or all missing
            unitMean = Double.NaN;
        } else {
            unitMean = this.unitSum / total;
        }

        return unitMean;
    }

    public double getMissingRate() {
        long total = getTotalInstCnt();
        double missingInstCnt = getTotalMissingCnt();
        return ((total != 0) ? missingInstCnt / total : 0.0);
    }

    public long getTotalInstCnt() {
        long total = 0;
        for ( int i = 0; i < binLen + 1; i ++) {
            total = total + this.positiveCounter[i] + this.negativeCounter[i];
        }
        return total;
    }

    protected long getTotalMissingCnt() {
        return this.positiveCounter[binLen] + this.negativeCounter[binLen];
    }

    public ColumnMetrics getDistMetrics() {
        return ColumnStatsCalculator.calculateColumnMetrics(this.negativeCounter, this.positiveCounter);
    }
}
