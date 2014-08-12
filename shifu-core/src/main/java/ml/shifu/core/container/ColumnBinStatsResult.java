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

import java.util.List;

/**
 * ColumnNumStatsResult class is stats collection for Column
 * If the Column type is categorical, the max/min field will be null
 * <p/>
 * ks/iv will be used for variable selection
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnBinStatsResult extends ColumnDerivedResult {


    private Double ks;
    private Double iv;
    private List<Double> binWoe;
    private List<Integer> binAvgScore;

    public List<Integer> getBinAvgScore() {
        return binAvgScore;
    }

    public void setBinAvgScore(List<Integer> binAvgScore) {
        this.binAvgScore = binAvgScore;
    }

    public List<Double> getBinWoe() {
        return binWoe;
    }

    public void setBinWoe(List<Double> binWoe) {
        this.binWoe = binWoe;
    }

    public Double getKs() {
        return ks;
    }

    public void setKs(Double ks) {
        this.ks = ks;
    }

    public Double getIv() {
        return iv;
    }

    public void setIv(Double iv) {
        this.iv = iv;
    }


}
