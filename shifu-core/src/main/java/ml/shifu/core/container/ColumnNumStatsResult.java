/**
 * Copyright [2012-2013] eBay Software Foundation
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
package ml.shifu.core.container;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * ColumnNumStatsResult class is stats collection for Column
 * If the Column type is categorical, the max/min field will be null
 * <p/>
 * ks/iv will be used for variable selection
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnNumStatsResult extends ColumnDerivedResult {

    private Double max;
    private Double min;
    private Double mean;
    private Double median;
    private Double stdDev;


    public Double getMax() {
        return max;
    }

    public void setMax(Double max) {
        this.max = max;
    }

    public Double getMin() {
        return min;
    }

    public void setMin(Double min) {
        this.min = min;
    }

    public Double getMean() {
        return mean;
    }

    public void setMean(Double mean) {
        this.mean = mean;
    }

    public Double getStdDev() {
        return stdDev;
    }

    public void setStdDev(Double stdDev) {
        this.stdDev = stdDev;
    }


    public Double getMedian() {
        return median;
    }

    public void setMedian(Double median) {
        this.median = median;
    }
}
