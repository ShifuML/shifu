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
package ml.shifu.shifu.container.obj;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import ml.shifu.shifu.util.Constants;

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

    /**
     * ---------------------------------------------------------------------------
     * <p/>
     * Auto-Gen methods
     * <p/>
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


    /**
     * ---------------------------------------------------------------------------
     * 
     * Capsulated methods for easy usage
     * 
     * ---------------------------------------------------------------------------
     */

    /**
     * @return
     */
    @JsonIgnore
    public boolean isTarget() {
        return ColumnFlag.Target.equals(columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isCandidate() {
        return !ColumnFlag.ForceRemove.equals(columnFlag) && !ColumnFlag.Meta.equals(columnFlag)
                && !ColumnFlag.Target.equals(columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isNumerical() {
        return columnType == ColumnType.N;
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isCategorical() {
        return columnType == ColumnType.C;
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isMeta() {
        return ColumnFlag.Meta == (columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isForceRemove() {
        return ColumnFlag.ForceRemove == (columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isForceSelect() {
        return ColumnFlag.ForceSelect == (columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public int getBinLength() {
        return columnBinning.getLength();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Double> getBinBoundary() {
        return columnBinning.getBinBoundary();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<String> getBinCategory() {
        return columnBinning.getBinCategory();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Integer> getBinCountNeg() {
        return columnBinning.getBinCountNeg();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Integer> getBinCountPos() {
        return columnBinning.getBinCountPos();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Double> getBinPosRate() {
        return columnBinning.getBinPosRate();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Integer> getBinAvgScore() {
        return columnBinning.getBinAvgScore();
    }

    /**
     * @param length
     */
    public void setBinLength(int length) {
        columnBinning.setLength(length);
    }

    /**
     * @param binBoundary
     */
    public void setBinBoundary(List<Double> binBoundary) {
        columnBinning.setBinBoundary(binBoundary);
        columnBinning.setLength(binBoundary.size());
    }

    /**
     * @param binCategory
     */
    public void setBinCategory(List<String> binCategory) {
        columnBinning.setBinCategory(binCategory);
        columnBinning.setLength(binCategory.size());
    }

    /**
     * @param binCountNeg
     */
    public void setBinCountNeg(List<Integer> binCountNeg) {
        columnBinning.setBinCountNeg(binCountNeg);
    }

    /**
     * @param binCountPos
     */
    public void setBinCountPos(List<Integer> binCountPos) {
        columnBinning.setBinCountPos(binCountPos);
    }

    /**
     * @param binPosCaseRate
     */
    public void setBinPosCaseRate(List<Double> binPosRate) {
        columnBinning.setBinPosRate(binPosRate);
    }

    /**
     * @param binAvgScore
     */
    public void setBinAvgScore(List<Integer> binAvgScore) {
        columnBinning.setBinAvgScore(binAvgScore);
    }

    /**
     * @return
     */
    @JsonIgnore
    public Double getKs() {
        return columnStats.getKs();
    }

    /**
     * @return
     */
    @JsonIgnore
    public Double getIv() {
        return columnStats.getIv();
    }

    /**
     * @return
     */
    @JsonIgnore
    public Double getMean() {
        return columnStats.getMean();
    }

    /**
     * @return
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
     * @param ks
     */
    public void setKs(double ks) {
        columnStats.setKs(ks);
    }

    /**
     * @param iv
     */
    public void setIv(double iv) {
        columnStats.setIv(iv);
    }

    /**
     * @param max
     */
    public void setMax(Double max) {
        columnStats.setMax(max);
    }

    /**
     * @param min
     */
    public void setMin(Double min) {
        columnStats.setMin(min);
    }

    /**
     * @param mean
     */
    public void setMean(Double mean) {
        columnStats.setMean(mean);
    }

    /**
     * @param stdDev
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

    /**
     * ColumnConfigComparator class
     */
    public static class ColumnConfigComparator implements Comparator<ColumnConfig> {
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
}
