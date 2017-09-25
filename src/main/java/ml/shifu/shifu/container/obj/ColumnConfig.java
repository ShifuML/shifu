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

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;

import ml.shifu.shifu.util.Constants;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * ColumnConfig class record the basic information for column in data. Almost all information in ColumnConfig is
 * generated automatically, user should avoid to change it manually, unless understanding the meaning of the changes.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnConfig {

    // add weight column and weight column is treated the same as meta
    public static enum ColumnFlag {
        ForceSelect, ForceRemove, Candidate, Meta, Target, Weight
    }

    /**
     * Column index, start from 0 to length-1
     */
    private Integer columnNum;

    /**
     * Column name read from header file or first line of csv file.
     */
    private String columnName;

    /**
     * Version of current ColumnConfig.json
     */
    private String version = Constants.version;

    /**
     * Numerical or Categorical feature.
     */
    private ColumnType columnType = ColumnType.N;

    /**
     * Meta, target, weight, force-select or force-remove columns
     */
    private ColumnFlag columnFlag = null;

    /**
     * If column is final selected, if set to finalSelect for {@link #columnFlag}, no matter what quality it is, such
     * column will be final selected and {@link #finalSelect} is set to true.
     * 
     * <p>
     * Only {@link #finalSelect} is determined as final training column.
     */
    private Boolean finalSelect = Boolean.FALSE;

    // in some hybrid threshold, some values like -100 is also category, try to set a threshold like 0, if value<0 would
    // be treated as categorical features, if not set, by default threshold is min double value, which means all value <
    // min can be set to categorical, which is all double value is not categorical, only works in Hybrid column type
    private Double hybridThreshold = Double.MIN_VALUE;

    /**
     * Column stats info
     */
    private ColumnStats columnStats = new ColumnStats();

    /**
     * Column binning info
     */
    private ColumnBinning columnBinning = new ColumnBinning();

    // /**
    // * Correlation array list
    // */
    // private double[] corrArray;

    /**
     * Sample values of such column.
     */
    private List<String> sampleValues;

    /*
     * ---------------------------------------------------------------------------
     * Auto-Gen methods
     * ---------------------------------------------------------------------------
     */

    public Integer getColumnNum() {
        return columnNum;
    }

    public void setColumnNum(Integer columnNum) {
        this.columnNum = columnNum;
    }

    public String getColumnName() {
        return columnName;
    }

    public ColumnType getColumnType() {
        return columnType;
    }

    public void setColumnType(ColumnType columnType) {
        this.columnType = columnType;
    }

    public ColumnFlag getColumnFlag() {
        return columnFlag;
    }

    public void setColumnFlag(ColumnFlag columnFlag) {
        this.columnFlag = columnFlag;
    }

    public Boolean isFinalSelect() {
        return finalSelect;
    }

    public void setFinalSelect(Boolean finalSelect) {
        this.finalSelect = finalSelect;
    }

    public void setColumnName(String columnName) {
        this.columnName = columnName;
    }

    public ColumnStats getColumnStats() {
        return columnStats;
    }

    public void setColumnStats(ColumnStats columnStats) {
        this.columnStats = columnStats;
    }

    public ColumnBinning getColumnBinning() {
        return columnBinning;
    }

    public void setColumnBinning(ColumnBinning columnBinning) {
        this.columnBinning = columnBinning;
    }

    /*
     * ---------------------------------------------------------------------------
     * 
     * Capsulated methods for easy usage
     * 
     * ---------------------------------------------------------------------------
     */

    @JsonIgnore
    public boolean isWeight() {
        return ColumnFlag.Weight == columnFlag;
    }

    @JsonIgnore
    public boolean isTarget() {
        return ColumnFlag.Target.equals(columnFlag);
    }

/*    @JsonIgnore
    public boolean isCandidate() {
        return ColumnFlag.Candidate.equals(columnFlag)
                || (!isForceRemove() && !isMeta() && !isTarget());
    }*/

    @JsonIgnore
    public boolean isCandidate(boolean hasCandidate) {
        return ( hasCandidate
                ? ColumnFlag.Candidate.equals(columnFlag) : (!isForceRemove() && !isMeta() && !isTarget()));
    }

    @JsonIgnore
    public boolean isNumerical() {
        // hybrid major is a numerical column but missing value is not target
        return columnType == ColumnType.N || columnType == ColumnType.H;
    }

    @JsonIgnore
    public boolean isCategorical() {
        return columnType == ColumnType.C;
    }

    @JsonIgnore
    public boolean isHybrid() {
        return columnType == ColumnType.H;
    }

    // weigt column is also treated as meta column
    @JsonIgnore
    public boolean isMeta() {
        return ColumnFlag.Meta == columnFlag || ColumnFlag.Weight == columnFlag;
    }

    @JsonIgnore
    public boolean isForceRemove() {
        return ColumnFlag.ForceRemove == (columnFlag);
    }

    @JsonIgnore
    public boolean isForceSelect() {
        return ColumnFlag.ForceSelect == (columnFlag);
    }

    @JsonIgnore
    public int getBinLength() {
        return columnBinning.getLength();
    }

    @JsonIgnore
    public List<Double> getBinBoundary() {
        return columnBinning.getBinBoundary();
    }

    @JsonIgnore
    public List<String> getBinCategory() {
        return columnBinning.getBinCategory();
    }

    @JsonIgnore
    public List<Integer> getBinCountNeg() {
        return columnBinning.getBinCountNeg();
    }

    @JsonIgnore
    public List<Integer> getBinCountPos() {
        return columnBinning.getBinCountPos();
    }

    @JsonIgnore
    public List<Double> getBinPosRate() {
        return columnBinning.getBinPosRate();
    }

    @JsonIgnore
    public List<Integer> getBinAvgScore() {
        return columnBinning.getBinAvgScore();
    }

    @JsonIgnore
    public List<Double> getBinCountWoe() {
        return columnBinning.getBinCountWoe();
    }

    @JsonIgnore
    public List<Double> getBinWeightedWoe() {
        return columnBinning.getBinWeightedWoe();
    }

    public void setBinLength(int length) {
        columnBinning.setLength(length);
    }

    public void setBinBoundary(List<Double> binBoundary) {
        columnBinning.setBinBoundary(binBoundary);
        columnBinning.setLength(binBoundary == null ? 0 : binBoundary.size());
    }

    public void setBinCategory(List<String> binCategory) {
        columnBinning.setBinCategory(binCategory);
        columnBinning.setLength(binCategory == null ? 0 : binCategory.size());
    }

    public void setBinCountNeg(List<Integer> binCountNeg) {
        columnBinning.setBinCountNeg(binCountNeg);
    }

    public void setBinCountPos(List<Integer> binCountPos) {
        columnBinning.setBinCountPos(binCountPos);
    }

    public void setBinPosCaseRate(List<Double> binPosRate) {
        columnBinning.setBinPosRate(binPosRate);
    }

    public void setBinAvgScore(List<Integer> binAvgScore) {
        columnBinning.setBinAvgScore(binAvgScore);
    }

    @JsonIgnore
    public Double getKs() {
        return columnStats.getKs();
    }

    @JsonIgnore
    public Double getIv() {
        return columnStats.getIv();
    }

    @JsonIgnore
    public Double getMean() {
        return columnStats.getMean();
    }

    @JsonIgnore
    public Double getStdDev() {
        return columnStats.getStdDev();
    }

    @JsonIgnore
    public Double getMedian() {
        return columnStats.getMedian();
    }

    @JsonIgnore
    public Long getMissingCount() {
        return columnStats.getMissingCount();
    }

    @JsonIgnore
    public Long getTotalCount() {
        return columnStats.getTotalCount();
    }

    @JsonIgnore
    public Double getMissingPercentage() {
        return columnStats.getMissingPercentage();
    }

    public void setKs(double ks) {
        columnStats.setKs(ks);
    }

    public void setIv(double iv) {
        columnStats.setIv(iv);
    }

    public void setMax(Double max) {
        columnStats.setMax(max);
    }

    public void setMin(Double min) {
        columnStats.setMin(min);
    }

    public void setMean(Double mean) {
        columnStats.setMean(mean);
    }

    public void setStdDev(Double stdDev) {
        columnStats.setStdDev(stdDev);
    }

    @JsonIgnore
    public void setMedian(Double median) {
        columnStats.setMedian(median);
    }

    @JsonIgnore
    public void setMissingCnt(Long cnt) {
        columnStats.setMissingCount(cnt);
    }

    @JsonIgnore
    public void setTotalCount(Long cnt) {
        columnStats.setTotalCount(cnt);
    }

    @JsonIgnore
    public void setMissingPercentage(Double missingPercentage) {
        columnStats.setMissingPercentage(missingPercentage);
    }

    @JsonIgnore
    public List<Double> getBinWeightedNeg() {
        return this.columnBinning.getBinWeightedNeg();
    }

    @JsonIgnore
    public List<Double> getBinWeightedPos() {
        return this.columnBinning.getBinWeightedPos();
    }

    @JsonIgnore
    public void setBinWeightedNeg(List<Double> binList) {
        this.columnBinning.setBinWeightedNeg(binList);
    }

    @JsonIgnore
    public void setBinWeightedPos(List<Double> binList) {
        this.columnBinning.setBinWeightedPos(binList);
    }

    /**
     * @return the version
     */
    public String getVersion() {
        return version;
    }

    /**
     * @param version
     *            the version to set
     */
    public void setVersion(String version) {
        this.version = version;
    }

    @JsonIgnore
    public void setPSI(Double psi) {
        this.columnStats.setPsi(psi);
    }

    @JsonIgnore
    public Double getPSI() {
        return this.columnStats.getPsi();
    }

    @JsonIgnore
    public List<String> getUnitStats() {
        return this.columnStats.getUnitStats();
    }

    @JsonIgnore
    public void setUnitStats(List<String> unitStats) {
        this.columnStats.setUnitStats(unitStats);
    }

    // /**
    // * @return the corrArray
    // */
    // public double[] getCorrArray() {
    // return corrArray;
    // }
    //
    // /**
    // * @param corrArray
    // * the corrArray to set
    // */
    // public void setCorrArray(double[] corrArray) {
    // this.corrArray = corrArray;
    // }

    /**
     * ColumnConfigComparator class
     */
    public static class ColumnConfigComparator implements Comparator<ColumnConfig>, Serializable {

        private static final long serialVersionUID = -1636776342389912951L;

        private String key;

        public ColumnConfigComparator(String key) {
            this.key = key;
        }

        public int compare(ColumnConfig a, ColumnConfig b) {
            if(key.equalsIgnoreCase("KS")) {
                return b.getKs().compareTo(a.getKs());
            } else {
                return b.getIv().compareTo(a.getIv());
            }
        }
    }

    @Override
    public ColumnConfig clone() {
        ColumnConfig other = new ColumnConfig();
        other.setColumnName(columnName);
        other.setColumnNum(columnNum);
        other.setVersion(version);
        other.setColumnType(columnType);
        other.setColumnFlag(columnFlag);
        other.setFinalSelect(finalSelect);
        other.setColumnStats(columnStats);
        other.setColumnBinning(columnBinning);
        // other.setCorrArray(corrArray);
        return other;
    }

    /**
     * @return the sampleValues
     */
    public List<String> getSampleValues() {
        return sampleValues;
    }

    /**
     * @param sampleValues
     *            the sampleValues to set
     */
    public void setSampleValues(List<String> sampleValues) {
        this.sampleValues = sampleValues;
    }

    /**
     * @return the hybridThreshold
     */
    public Double getHybridThreshold() {
        return hybridThreshold;
    }

    /**
     * @param hybridThreshold
     *            the hybridThreshold to set
     */
    public void setHybridThreshold(Double hybridThreshold) {
        this.hybridThreshold = hybridThreshold;
    }
}
