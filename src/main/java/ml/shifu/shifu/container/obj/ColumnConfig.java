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
    private ColumnRawStatsResult columnRawStatsResult = new ColumnRawStatsResult();

    private ColumnNumStatsResult columnNumStatsResult = new ColumnNumStatsResult();
    private ColumnBinningResult columnBinningResult = new ColumnBinningResult();

    public ColumnBinStatsResult getColumnBinStatsResult() {
        return columnBinStatsResult;
    }

    public void setColumnBinStatsResult(ColumnBinStatsResult columnBinStatsResult) {
        this.columnBinStatsResult = columnBinStatsResult;
    }

    private ColumnBinStatsResult columnBinStatsResult = new ColumnBinStatsResult();


    private ColumnControl columnControl = new ColumnControl();

    /**
     * ---------------------------------------------------------------------------
     *
     * 					Auto-Gen methods
     *
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

    public ColumnRawStatsResult getColumnRawStatsResult() {
        return columnRawStatsResult;
    }

    public void setColumnRawStatsResult(ColumnRawStatsResult columnRawStatsResult) {
        this.columnRawStatsResult = columnRawStatsResult;
    }

    public ColumnNumStatsResult getColumnNumStatsResult() {
        return columnNumStatsResult;
    }

    public void setColumnNumStatsResult(ColumnNumStatsResult columnNumStatsResult) {
        this.columnNumStatsResult = columnNumStatsResult;
    }

    public ColumnBinningResult getColumnBinningResult() {
        return columnBinningResult;
    }

    public void setColumnBinningResult(ColumnBinningResult columnBinningResult) {
        this.columnBinningResult = columnBinningResult;
    }

    public ColumnControl getColumnControl() {
        return columnControl;
    }

    public void setColumnControl(ColumnControl columnControl) {
        this.columnControl = columnControl;
    }

    /**
     * ---------------------------------------------------------------------------
     *
     * 					Capsulated methods for easy usage
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
        return !ColumnFlag.ForceRemove.equals(columnFlag)
                && !ColumnFlag.Meta.equals(columnFlag)
                && !ColumnFlag.Target.equals(columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isNumerical() {
        return columnType.equals(ColumnType.N);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isCategorical() {
        if (columnType == null) {
            return false;
        }
        return columnType.equals(ColumnType.C);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isMeta() {
        return ColumnFlag.Meta.equals(columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isForceRemove() {
        return ColumnFlag.ForceRemove.equals(columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public boolean isForceSelect() {
        return ColumnFlag.ForceSelect.equals(columnFlag);
    }

    /**
     * @return
     */
    @JsonIgnore
    public int getBinLength() {
        return columnBinningResult.getLength();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Double> getBinBoundary() {
        return columnBinningResult.getBinBoundary();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<String> getBinCategory() {
        return columnBinningResult.getBinCategory();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Integer> getBinCountNeg() {
        return columnBinningResult.getBinCountNeg();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Integer> getBinCountPos() {
        return columnBinningResult.getBinCountPos();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Double> getBinPosRate() {
        return columnBinningResult.getBinPosRate();
    }

    /**
     * @return
     */
    @JsonIgnore
    public List<Integer> getBinAvgScore() {
        return columnBinStatsResult.getBinAvgScore();
    }

    /**
     * @param length
     */
    public void setBinLength(int length) {
        columnBinningResult.setLength(length);
    }

    /**
     * @param binBoundary
     */
    public void setBinBoundary(List<Double> binBoundary) {
        columnBinningResult.setBinBoundary(binBoundary);
        columnBinningResult.setLength(binBoundary.size());
    }

    /**
     * @param binCategory
     */
    public void setBinCategory(List<String> binCategory) {
        columnBinningResult.setBinCategory(binCategory);
        columnBinningResult.setLength(binCategory.size());
    }


    /**
     * @param binCountNeg
     */
    public void setBinCountNeg(List<Integer> binCountNeg) {
        columnBinningResult.setBinCountNeg(binCountNeg);
    }

    /**
     * @param binCountPos
     */
    public void setBinCountPos(List<Integer> binCountPos) {
        columnBinningResult.setBinCountPos(binCountPos);
    }

    /**
     * @param binPosRate
     */
    public void setBinPosCaseRate(List<Double> binPosRate) {
        columnBinningResult.setBinPosRate(binPosRate);
    }

    /**
     * @param binAvgScore
     */
    public void setBinAvgScore(List<Integer> binAvgScore) {
        columnBinStatsResult.setBinAvgScore(binAvgScore);
    }

    /**
     * @return
     */
    @JsonIgnore
    public Double getMean() {
        return columnNumStatsResult.getMean();
    }

    /**
     * @return
     */
    @JsonIgnore
    public Double getStdDev() {
        return columnNumStatsResult.getStdDev();
    }

    @JsonIgnore
    public Double getMedian(){
        return columnNumStatsResult.getMedian();
    }


    /**
     * @param max
     */
    public void setMax(Double max) {
        columnNumStatsResult.setMax(max);
    }

    /**
     * @param min
     */
    public void setMin(Double min) {
        columnNumStatsResult.setMin(min);
    }

    /**
     * @param mean
     */
    public void setMean(Double mean) {
        columnNumStatsResult.setMean(mean);
    }

    /**
     * @param stdDev
     */
    public void setStdDev(Double stdDev) {
        columnNumStatsResult.setStdDev(stdDev);
    }

    @JsonIgnore
    public void setMedian(Double median) {
        columnNumStatsResult.setMedian(median);
    }



    @JsonIgnore
    public List<Double> getBinWeightedNeg(){
        return this.columnBinningResult.getBinWeightedNeg();
    }

    @JsonIgnore
    public List<Double> getBinWeightedPos(){
        return this.columnBinningResult.getBinWeightedPos();
    }

    @JsonIgnore
    public void setBinWeightedNeg(List<Double> binList){
        this.columnBinningResult.setBinWeightedNeg(binList);
    }

    @JsonIgnore
    public void setBinWeightedPos(List<Double> binList){
        this.columnBinningResult.setBinWeightedPos(binList);
    }

    //@JsonIgnore

    /**
     * @return the version
     */
    public String getVersion() {
        return version;
    }

    /**
     * @param version the version to set
     */
    public void setVersion(String version) {
        this.version = version;
    }

    /**
     *
     * ColumnConfigComparator class
     *
     */

    public static class ColumnConfigComparator implements Comparator<ColumnConfig> {
        private String key;

        public ColumnConfigComparator(String key) {
            this.key = key;
        }

        public int compare(ColumnConfig a, ColumnConfig b) {
            if (key.equalsIgnoreCase("KS")) {
                return b.getColumnBinStatsResult().getKs().compareTo(a.getColumnBinStatsResult().getKs());
            } else {
                return b.getColumnBinStatsResult().getIv().compareTo(a.getColumnBinStatsResult().getIv());
            }
        }
    }
}
