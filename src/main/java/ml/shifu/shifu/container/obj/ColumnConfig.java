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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import ml.shifu.shifu.util.Constants;

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;

/**
 * ColumnConfig class record the basic information for column in data.
 * Almost all information in ColumnConfig is generated automatically, user should
 * avoid to change it manually, unless understanding the meaning of the changes.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnConfig {

    public static enum ColumnFlag {
        ForceSelect, ForceRemove, Meta, Target
    }

    public static enum ColumnType {
        A, N, C
    }

    // basic info
    private Integer columnNum;
    private String columnName;

    // version
    private String version = Constants.version;

    // column type
    private ColumnType columnType = ColumnType.N;
    private ColumnFlag columnFlag = null;
    private Boolean finalSelect = Boolean.FALSE;

    // column detail
    private ColumnStats columnStats = new ColumnStats();
    private ColumnBinning columnBinning = new ColumnBinning();

    private List<Double> corrArray;
    
    private List<String> sampleValues;

    /**
     * ---------------------------------------------------------------------------
     * <p>
     * Auto-Gen methods
     * </p>
     * ---------------------------------------------------------------------------
     */

    /**
     * @return columnNum
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

    /**
     * ---------------------------------------------------------------------------
     * 
     * Capsulated methods for easy usage
     * 
     * ---------------------------------------------------------------------------
     */

    /**
     * @return column is target or not
     */
    @JsonIgnore
    public boolean isTarget() {
        return ColumnFlag.Target.equals(columnFlag);
    }

    /**
     * @return column is candidate variable or not
     */
    @JsonIgnore
    public boolean isCandidate() {
        return !ColumnFlag.ForceRemove.equals(columnFlag) && !ColumnFlag.Meta.equals(columnFlag)
                && !ColumnFlag.Target.equals(columnFlag);
    }

    /**
     * @return column is numerical variable or not
     */
    @JsonIgnore
    public boolean isNumerical() {
        return columnType == ColumnType.N;
    }

    /**
     * @return column is categorical or not
     */
    @JsonIgnore
    public boolean isCategorical() {
        return columnType == ColumnType.C;
    }

    /**
     * @return column is meta or not
     */
    @JsonIgnore
    public boolean isMeta() {
        return ColumnFlag.Meta == (columnFlag);
    }

    /**
     * @return column is force-removed or not
     */
    @JsonIgnore
    public boolean isForceRemove() {
        return ColumnFlag.ForceRemove == (columnFlag);
    }

    /**
     * @return column is force-selected or not
     */
    @JsonIgnore
    public boolean isForceSelect() {
        return ColumnFlag.ForceSelect == (columnFlag);
    }

    /**
     * @return bin number
     */
    @JsonIgnore
    public int getBinLength() {
        return columnBinning.getLength();
    }

    /**
     * @return bin boundary for numerical variable
     */
    @JsonIgnore
    public List<Double> getBinBoundary() {
        return columnBinning.getBinBoundary();
    }

    /**
     * @return bin category for categorical variable
     */
    @JsonIgnore
    public List<String> getBinCategory() {
        return columnBinning.getBinCategory();
    }

    /**
     * @return negative instance count on each bin
     */
    @JsonIgnore
    public List<Integer> getBinCountNeg() {
        return columnBinning.getBinCountNeg();
    }

    /**
     * @return positive instance count on each bin
     */
    @JsonIgnore
    public List<Integer> getBinCountPos() {
        return columnBinning.getBinCountPos();
    }

    /**
     * @return positive rate on each bin
     */
    @JsonIgnore
    public List<Double> getBinPosRate() {
        return columnBinning.getBinPosRate();
    }

    /**
     * @return average score on each bin
     */
    @JsonIgnore
    public List<Integer> getBinAvgScore() {
        return columnBinning.getBinAvgScore();
    }

    /**
     * @return count WOE on each bin
     */
    @JsonIgnore
    public List<Double> getBinCountWoe() {
        return columnBinning.getBinCountWoe();
    }

    /**
     * @return weighted WOE on each bin
     */
    @JsonIgnore
    public List<Double> getBinWeightedWoe() {
        return columnBinning.getBinWeightedWoe();
    }

    /**
     * @param length - length to set
     */
    public void setBinLength(int length) {
        columnBinning.setLength(length);
    }

    /**
     * @param binBoundary - binBoundary to set
     */
    public void setBinBoundary(List<Double> binBoundary) {
        columnBinning.setBinBoundary(binBoundary);
        columnBinning.setLength(binBoundary.size());
    }

    /**
     * @param binCategory - binCategory to set
     */
    public void setBinCategory(List<String> binCategory) {
        columnBinning.setBinCategory(binCategory);
        columnBinning.setLength(binCategory.size());
    }

    /**
     * @param binCountNeg - binCountNeg to set
     */
    public void setBinCountNeg(List<Integer> binCountNeg) {
        columnBinning.setBinCountNeg(binCountNeg);
    }

    /**
     * @param binCountPos - binCountPos to set
     */
    public void setBinCountPos(List<Integer> binCountPos) {
        columnBinning.setBinCountPos(binCountPos);
    }

    /**
     * @param binPosRate - binPosRate to set
     */
    public void setBinPosCaseRate(List<Double> binPosRate) {
        columnBinning.setBinPosRate(binPosRate);
    }

    /**
     * @param binAvgScore - binAvgScore to set
     */
    public void setBinAvgScore(List<Integer> binAvgScore) {
        columnBinning.setBinAvgScore(binAvgScore);
    }

    /**
     * @return ks
     */
    @JsonIgnore
    public Double getKs() {
        return columnStats.getKs();
    }

    /**
     * @return iv
     */
    @JsonIgnore
    public Double getIv() {
        return columnStats.getIv();
    }

    /**
     * @return mean
     */
    @JsonIgnore
    public Double getMean() {
        return columnStats.getMean();
    }

    /**
     * @return stdDev
     */
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

    /**
     * @param ks - ks to set
     */
    public void setKs(double ks) {
        columnStats.setKs(ks);
    }

    /**
     * @param iv - iv to set
     */
    public void setIv(double iv) {
        columnStats.setIv(iv);
    }

    /**
     * @param max - max to set
     */
    public void setMax(Double max) {
        columnStats.setMax(max);
    }

    /**
     * @param min - min to set
     */
    public void setMin(Double min) {
        columnStats.setMin(min);
    }

    /**
     * @param mean - mean to set
     */
    public void setMean(Double mean) {
        columnStats.setMean(mean);
    }

    /**
     * @param stdDev - stdDev to set
     */
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

    /**
     * @return the corrArray
     */
    public List<Double> getCorrArray() {
        return corrArray;
    }

    /**
     * @param corrArray the corrArray to set
     */
    public void setCorrArray(List<Double> corrArray) {
        this.corrArray = corrArray;
    }

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
        other.setCorrArray(corrArray);
        return other;
    }

    /**
     * @return the sampleValues
     */
    public List<String> getSampleValues() {
        return sampleValues;
    }

    /**
     * @param sampleValues the sampleValues to set
     */
    public void setSampleValues(List<String> sampleValues) {
        this.sampleValues = sampleValues;
    }
}
