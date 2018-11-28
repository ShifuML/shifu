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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * ColumnStats class is stats collection for Column; If the Column type is categorical, the max/min field will be null.
 * 
 * <p>
 * 'ks/iv' will be used for variable selection.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnStats {
    /**
     * Max value of such column
     */
    private Double max;

    /**
     * Min value of such column
     */
    private Double min;

    /**
     * Mean value of such column
     */
    private Double mean;

    /**
     * Median value of such column (now it is the same as mean)
     */
    private Double median;

    /**
     * the 25 percentile value of the column
     */
    private Double p25th;

    /**
     * the 75 percentile value of the column
     */
    private Double p75th;

    /**
     * Total count for such column
     */
    private Long totalCount;

    /**
     * Distinct count (for categorical feature it is cardity) Missing count are not included
     */
    private Long distinctCount;

    /**
     * Missing count, missing data are data in missingDataList
     */
    private Long missingCount;

    /**
     * Number of valid numeric count
     */
    private Long validNumCount;

    /**
     * Stand deviation
     */
    private Double stdDev;

    /**
     * Missing percentage (missingCount/totalCount)
     */
    private Double missingPercentage;

    /**
     * WOE value for such column
     */
    private Double woe;

    /**
     * KS value
     */
    private Double ks;

    /**
     * Information value
     */
    private Double iv;

    /**
     * Weighted KS value
     */
    private Double weightedKs;

    /**
     * Weighted IV value
     */
    private Double weightedIv;

    /**
     * Weighted woe value
     */
    private Double weightedWoe;

    /**
     * Skewness of such column
     */
    private Double skewness;

    /**
     * Kurtosis of such column
     */
    private Double kurtosis;

    /**
     * PSI value of such column
     */
    private Double psi;

    /**
     * Unit stats information
     */
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

    public Double get25th() {
        return this.p25th;
    }

    public void set25th(Double p25th) {
        this.p25th = p25th;
    }

    public Double get75th() {
        return this.p75th;
    }

    public void set75th(Double p75th) {
        this.p75th = p75th;
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
     * @param skewness
     *            the skewness to set
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
     * @param kurtosis
     *            the kurtosis to set
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
     * @param psi
     *            the PSI to set
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
     * @param unitStats
     *            the unitStats
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
     * @param validNumCount
     *            the validNumCount to set
     */
    public void setValidNumCount(Long validNumCount) {
        this.validNumCount = validNumCount;
    }

    /**
     * Write columnStatus to output stream
     * @param output output stream
     * @throws IOException io exception
     */
    public void write(DataOutputStream output) throws IOException {
        writeWrapperValue(output, max);
        writeWrapperValue(output, min);
        writeWrapperValue(output, mean);
        writeWrapperValue(output, median);
        writeWrapperValue(output, p25th);
        writeWrapperValue(output, p75th);
        writeWrapperValue(output, totalCount);
        writeWrapperValue(output, distinctCount);
        writeWrapperValue(output, missingCount);
        writeWrapperValue(output, validNumCount);
        writeWrapperValue(output, stdDev);
        writeWrapperValue(output, missingPercentage);
        writeWrapperValue(output, woe);
        writeWrapperValue(output, ks);
        writeWrapperValue(output, iv);
        writeWrapperValue(output, weightedKs);
        writeWrapperValue(output, weightedIv);
        writeWrapperValue(output, weightedWoe);
        writeWrapperValue(output, skewness);
        writeWrapperValue(output, kurtosis);
        writeWrapperValue(output, psi);
        if (unitStats == null || unitStats.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(unitStats.size());
            for (String unitStat : unitStats) {
                output.writeUTF(unitStat);
            }
        }
    }

    /**
     * Read columnstats from input stream
     * @param input input stream
     * @throws IOException io exception
     */
    public void read(DataInputStream input) throws IOException {
        max = readWrapperValue(input, Double.class);
        min = readWrapperValue(input, Double.class);
        mean = readWrapperValue(input, Double.class);
        median = readWrapperValue(input, Double.class);
        p25th = readWrapperValue(input, Double.class);
        p75th = readWrapperValue(input, Double.class);
        totalCount = readWrapperValue(input, Long.class);
        distinctCount = readWrapperValue(input, Long.class);
        missingCount = readWrapperValue(input, Long.class);
        validNumCount = readWrapperValue(input, Long.class);
        stdDev = readWrapperValue(input, Double.class);
        missingPercentage = readWrapperValue(input, Double.class);
        woe = readWrapperValue(input, Double.class);
        ks = readWrapperValue(input, Double.class);
        iv = readWrapperValue(input, Double.class);
        weightedKs = readWrapperValue(input, Double.class);
        weightedIv = readWrapperValue(input, Double.class);
        weightedWoe = readWrapperValue(input, Double.class);
        skewness = readWrapperValue(input, Double.class);
        kurtosis = readWrapperValue(input, Double.class);
        psi = readWrapperValue(input, Double.class);
        int unitNum = input.readInt();
        if (unitNum != 0) {
            unitStats = new ArrayList<String>();
            for (int i = 0; i < unitNum; i++) {
                unitStats.add(input.readUTF());
            }
        }
    }

    private void writeWrapperValue(DataOutputStream output, Number field) throws IOException {
        if (field == null) {
            output.writeBoolean(true);
        } else {
            output.writeBoolean(false);
            if (field instanceof Double) {
                output.writeDouble(field.doubleValue());
            } else if (field instanceof Long) {
                output.writeLong(field.longValue());
            }
        }
    }

    private <T extends Number>T readWrapperValue(DataInputStream input, Class<T> clazz) throws IOException {
        boolean nullFlag = input.readBoolean();
        if (nullFlag) return null;
        else if (clazz == Double.class){
            return clazz.cast(input.readDouble());
        } else {
            return clazz.cast(input.readLong());
        }
    }
}
