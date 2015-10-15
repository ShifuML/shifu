/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.container.obj;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

/**
 * ColumnBinning class represents the information of BINNING. Usually the BINNING information will be used to calculate
 * KS/IV, or reason code in evaluation.
 * 
 * <p>
 * Please note for numerical variables, @binCategory will be null, but for categorical variables @binBoundary will be
 * null. The @binLength will equal size of @binBoundary or size of @binCategory.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnBinning {

    private Integer length = Integer.valueOf(0);

    /**
     * Works for numerical feature, then {@link #binCategory} is null.
     */
    private List<Double> binBoundary;

    /**
     * Works for categorical feature, it is all categories of such column
     */
    private List<String> binCategory;

    /**
     * Count of negative records in bins
     */
    private List<Integer> binCountNeg;

    /**
     * Count of positive records in bins
     */
    private List<Integer> binCountPos;

    /**
     * Positive rate in each bin
     */
    private List<Double> binPosRate;

    /**
     * Average score in each bin, this will be populated in posttrain step.
     */
    private List<Integer> binAvgScore;

    /**
     * Weighted negative value in each bin.
     */
    private List<Double> binWeightedNeg;

    /**
     * Weighted positive value in each bin.
     */
    private List<Double> binWeightedPos;

    /**
     * Woe value in each bin
     */
    private List<Double> binCountWoe;

    /**
     * Weighted woe value in each bin
     */
    private List<Double> binWeightedWoe;

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

    public List<Integer> getBinAvgScore() {
        return binAvgScore;
    }

    public void setBinAvgScore(List<Integer> binAvgScore) {
        this.binAvgScore = binAvgScore;
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

    /**
     * @return the binCountWoe
     */
    public List<Double> getBinCountWoe() {
        return binCountWoe;
    }

    /**
     * @param binCountWoe
     *            the binCountWoe to set
     */
    public void setBinCountWoe(List<Double> binCountWoe) {
        this.binCountWoe = binCountWoe;
    }

    /**
     * @return the binWeightedWoe
     */
    public List<Double> getBinWeightedWoe() {
        return binWeightedWoe;
    }

    /**
     * @param binWeightedWoe
     *            the binWeightedWoe to set
     */
    public void setBinWeightedWoe(List<Double> binWeightedWoe) {
        this.binWeightedWoe = binWeightedWoe;
    }

}
