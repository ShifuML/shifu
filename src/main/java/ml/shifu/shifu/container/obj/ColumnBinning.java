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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
     * A map version of {@link #binCategory},
     */
    @JsonIgnore
    private Map<String, Integer> binCateMap;

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

    @JsonIgnore
    public Map<String, Integer> getBinCateMap() {
        return binCateMap;
    }

    @JsonIgnore
    public void setBinCateMap(Map<String, Integer> binCateMap) {
        this.binCateMap = binCateMap;
    }

    /**
     * Read columnbinning from input stream
     * @param output output stream
     * @throws IOException io exception
     */
    public void write(DataOutputStream output) throws IOException {
        output.writeInt(length);

        if (binBoundary == null || binBoundary.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binBoundary.size());
            for (Double binBound : binBoundary) {
                output.writeDouble(binBound);
            }
        }

        if (binCategory == null || binCategory.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binCategory.size());
            for (String binCat : binCategory) {
                output.writeUTF(binCat);
            }
        }

        if (binCateMap == null || binCateMap.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binCateMap.size());
            for (Map.Entry<String, Integer> binCateEntry : binCateMap.entrySet()) {
                output.writeUTF(binCateEntry.getKey());
                output.writeInt(binCateEntry.getValue());
            }
        }

        if (binCountNeg == null || binCountNeg.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binCountNeg.size());
            for (Integer binCountNegItem: binCountNeg) {
                output.writeInt(binCountNegItem);
            }
        }

        if (binCountPos == null || binCountPos.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binCountPos.size());
            for (Integer binCountPosItem: binCountPos) {
                output.writeInt(binCountPosItem);
            }
        }

        if (binPosRate == null || binPosRate.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binPosRate.size());
            for (Double binPosRateItem : binPosRate) {
                output.writeDouble(binPosRateItem);
            }
        }

        if (binAvgScore == null || binAvgScore.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binAvgScore.size());
            for (Integer binAvgScoreItem: binAvgScore) {
                output.writeInt(binAvgScoreItem);
            }
        }

        if (binWeightedNeg == null || binWeightedNeg.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binWeightedNeg.size());
            for (Double binWeightedNegItem : binWeightedNeg) {
                output.writeDouble(binWeightedNegItem);
            }
        }

        if (binWeightedPos == null || binWeightedPos.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binWeightedPos.size());
            for (Double binWeightedPosItem : binWeightedPos) {
                output.writeDouble(binWeightedPosItem);
            }
        }

        if (binCountWoe == null || binCountWoe.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binCountWoe.size());
            for (Double binCountWoeItem : binCountWoe) {
                output.writeDouble(binCountWoeItem);
            }
        }

        if (binWeightedWoe == null || binWeightedWoe.isEmpty()) {
            output.writeInt(0);
        } else {
            output.writeInt(binWeightedWoe.size());
            for (Double binWeightedWoeItem : binWeightedWoe) {
                output.writeDouble(binWeightedWoeItem);
            }
        }
    }

    /**
     * Read columnbinning from input stream
     * @param input input stream
     */
    public void read(DataInputStream input) throws IOException {
        length = input.readInt();

        int size = input.readInt();
        binBoundary = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binBoundary.add(input.readDouble());
        }

        size = input.readInt();
        binCategory = new ArrayList<String>();
        for (int i = 0; i < size; i++) {
            binCategory.add(input.readUTF());
        }

        size = input.readInt();
        binCateMap = new HashMap<String, Integer>();
        for (int i = 0; i < size; i++) {
            binCateMap.put(input.readUTF(), input.readInt());
        }

        size = input.readInt();
        binCountNeg = new ArrayList<Integer>();
        for (int i = 0; i < size; i++) {
            binCountNeg.add(input.readInt());
        }

        size = input.readInt();
        binCountPos = new ArrayList<Integer>();
        for (int i = 0; i < size; i++) {
            binCountPos.add(input.readInt());
        }

        size = input.readInt();
        binPosRate = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binPosRate.add(input.readDouble());
        }

        size = input.readInt();
        binAvgScore = new ArrayList<Integer>();
        for (int i = 0; i < size; i++) {
            binAvgScore.add(input.readInt());
        }

        size = input.readInt();
        binWeightedNeg = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binWeightedNeg.add(input.readDouble());
        }

        size = input.readInt();
        binWeightedPos = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binWeightedPos.add(input.readDouble());
        }

        size = input.readInt();
        binCountWoe = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binCountWoe.add(input.readDouble());
        }

        size = input.readInt();
        binWeightedPos = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binWeightedPos.add(input.readDouble());
        }

        size = input.readInt();
        binCountWoe = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binCountWoe.add(input.readDouble());
        }

        size = input.readInt();
        binWeightedWoe = new ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            binWeightedWoe.add(input.readDouble());
        }
    }

}
