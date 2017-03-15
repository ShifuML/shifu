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
 * ColumnStats class is stats collection for Column
 * If the Column type is categorical, the max/min field will be null
 * <p>
 * ks/iv will be used for variable selection
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnStats {

    private Double max;
    private Double min;
    private Double mean;
    private Double median;
    private Long totalCount;
    /**
     * Missing count are not included
     */
    private Long distinctCount;
    private Long missingCount;
    private Long validNumCount;
    private Double stdDev;
    private Double missingPercentage;
    private Double woe;
    private Double ks;
    private Double iv;
    private Double weightedKs;
    private Double weightedIv;
    private Double weightedWoe;

    private Double skewness;
    private Double kurtosis;

    private Double psi;
    private List<String> unitStats;

    /**
     * @return the weightedKs
     */
    public Double getWeightedKs() {
        return weightedKs;
    }

    /**
     * @param weightedKs
     *            the weightedKs to set
     */
    public void setWeightedKs(Double weightedKs) {
        this.weightedKs = weightedKs;
    }

    /**
     * @return the weightedIv
     */
    public Double getWeightedIv() {
        return weightedIv;
    }

    /**
     * @param weightedIv
     *            the weightedIv to set
     */
    public void setWeightedIv(Double weightedIv) {
        this.weightedIv = weightedIv;
    }

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

    public Double getMedian() {
        return median;
    }

    public void setMedian(Double median) {
        this.median = median;
    }

    public Long getTotalCount() {
        return totalCount;
    }

    public void setTotalCount(Long totalCount) {
        this.totalCount = totalCount;
    }

    public Long getMissingCount() {
        return missingCount;
    }

    public void setMissingCount(Long missingCount) {
        this.missingCount = missingCount;
    }

    public Double getMissingPercentage() {
        return missingPercentage;
    }

    public void setMissingPercentage(Double missingPercentage) {
        this.missingPercentage = missingPercentage;
    }

    /**
     * @return the woe
     */
    public Double getWoe() {
        return woe;
    }

    /**
     * @return the weightedWoe
     */
    public Double getWeightedWoe() {
        return weightedWoe;
    }

    /**
     * @param woe
     *            the woe to set
     */
    public void setWoe(Double woe) {
        this.woe = woe;
    }

    /**
     * @param weightedWoe
     *            the weightedWoe to set
     */
    public void setWeightedWoe(Double weightedWoe) {
        this.weightedWoe = weightedWoe;
    }

    /**
     * @return the distinctCount
     */
    public Long getDistinctCount() {
        return distinctCount;
    }

    /**
     * @param distinctCount
     *            the distinctCount to set
     */
    public void setDistinctCount(Long distinctCount) {
        this.distinctCount = distinctCount;
    }

    /**
     * @return the skewness
     */
    public Double getSkewness() {
        return skewness;
    }

    /**
     * @param skewness the skewness to set
     */
    public void setSkewness(Double skewness) {
        this.skewness = skewness;
    }

    /**
     * @return the kurtosis
     */
    public Double getKurtosis() {
        return kurtosis;
    }

    /**
     * @param kurtosis the kurtosis to set
     */
    public void setKurtosis(Double kurtosis) {
        this.kurtosis = kurtosis;
    }

    /**
     * @return the PSI
     */
    public Double getPsi() {
        return psi;
    }

    /**
     * @param psi the PSI to set
     */
    public void setPsi(Double psi) {
        this.psi = psi;
    }

    /**
     * @return return List of unit stats
     */
    public List<String> getUnitStats() {
        return unitStats;
    }

    /**
     *
     * @param unitStats the unitStats
     */
    public void setUnitStats(List<String> unitStats) {
        this.unitStats = unitStats;
    }

    /**
     * @return the validNumCount
     */
    public Long getValidNumCount() {
        return validNumCount;
    }

    /**
     * @param validNumCount the validNumCount to set
     */
    public void setValidNumCount(Long validNumCount) {
        this.validNumCount = validNumCount;
    }
}
