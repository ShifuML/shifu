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
 * ColumnBinningResult class represents the information of BINNING.
 * Usually the BINNING information will be used to calculate KS/IV, or reason code in evaluation.
 * <p/>
 * Please note for numerical variables, @binCategory will be null, but
 * for categorical variables @binBoundary will be null.
 * The @binLength will equal size of @binBoundary or size of @binCategory.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnBinningResult extends ColumnDerivedResult {

    private Integer length = 0;

    private List<Double> binBoundary;
    private List<String> binCategory;

    private List<Integer> binCountNeg;
    private List<Integer> binCountPos;
    private List<Double> binPosRate;


    private List<Double> binWeightedNeg;
    private List<Double> binWeightedPos;


    public Integer getLength() {
        return length;
    }

    public void setLength(Integer length) {
        this.length = length;
    }

    public List<Double> getBinBoundary() {
        return binBoundary;
    }

    public void setBinBoundary(List<Double> binBoundary) {
        this.binBoundary = binBoundary;
    }

    public List<String> getBinCategory() {
        return binCategory;
    }

    public void setBinCategory(List<String> binCategory) {
        this.binCategory = binCategory;
    }

    public List<Integer> getBinCountNeg() {
        return binCountNeg;
    }

    public void setBinCountNeg(List<Integer> binCountNeg) {
        this.binCountNeg = binCountNeg;
    }

    public List<Integer> getBinCountPos() {
        return binCountPos;
    }

    public void setBinCountPos(List<Integer> binCountPos) {
        this.binCountPos = binCountPos;
    }

    public List<Double> getBinPosRate() {
        return binPosRate;
    }

    public void setBinPosRate(List<Double> binPosRate) {
        this.binPosRate = binPosRate;
    }


    public List<Double> getBinWeightedNeg() {
        return binWeightedNeg;
    }

    public void setBinWeightedNeg(List<Double> binWeightedNeg) {
        this.binWeightedNeg = binWeightedNeg;
    }

    public List<Double> getBinWeightedPos() {
        return binWeightedPos;
    }

    public void setBinWeightedPos(List<Double> binWeightedPos) {
        this.binWeightedPos = binWeightedPos;
    }

}
