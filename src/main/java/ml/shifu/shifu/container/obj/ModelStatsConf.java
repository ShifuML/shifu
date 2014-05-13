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
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * ModelStatsConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelStatsConf {

    @JsonDeserialize(using = BinningMethodDeserializer.class)
    public static enum BinningMethod {
        EqualPositive, EqualTotal, EqualInterval
    }

    private Integer maxNumBin = Integer.valueOf(10);
    private BinningMethod binningMethod = BinningMethod.EqualPositive;
    private Double sampleRate = Double.valueOf(0.8);
    private Boolean sampleNegOnly = Boolean.FALSE;

    // don't open those options to user
    private Double numericalValueThreshold = Double.MAX_VALUE;
    private Boolean binningAutoTypeEnable = Boolean.FALSE;
    private Integer binningAutoTypeThreshold = Integer.valueOf(5);
    private Boolean binningMergeEnable = Boolean.TRUE;

    private String rawStatsCalculator = "ml.shifu.shifu.di.builtin.DefaultColumnRawStatsCalculator";
    private String numBinningCalculator = "ml.shifu.shifu.di.builtin.TotalPercentileColumnNumBinningCalculator";
    private String catBinningCalculator = "ml.shifu.shifu.di.builtin.SimpleColumnCatBinningCalculator";
    private String numStatsCalculator = "ml.shifu.shifu.di.builtin.DefaultColumnNumStatsCalculator";
    private String binStatsCalculator = "ml.shifu.shifu.di.builtin.DefaultColumnBinStatsCalculator";



    private String statsProcessor = "ml.shifu.shifu.di.builtin.DefaultStatsProcessor";


    public String getStatsProcessor() {
        return statsProcessor;
    }

    public void setStatsProcessor(String statsProcessor) {
        this.statsProcessor = statsProcessor;
    }


    public Integer getMaxNumBin() {
        return maxNumBin;
    }

    public void setMaxNumBin(Integer maxNumBin) {
        this.maxNumBin = maxNumBin;
    }

    @JsonIgnore
    public Double getNumericalValueThreshold() {
        return numericalValueThreshold;
    }

    public void setNumericalValueThreshold(Double numericalValueThreshold) {
        this.numericalValueThreshold = numericalValueThreshold;
    }

    @JsonIgnore
    public Boolean getBinningAutoTypeEnable() {
        return binningAutoTypeEnable;
    }

    public void setBinningAutoTypeEnable(Boolean binningAutoTypeEnable) {
        this.binningAutoTypeEnable = binningAutoTypeEnable;
    }

    @JsonIgnore
    public Integer getBinningAutoTypeThreshold() {
        return binningAutoTypeThreshold;
    }

    public void setBinningAutoTypeThreshold(Integer binningAutoTypeThreshold) {
        this.binningAutoTypeThreshold = binningAutoTypeThreshold;
    }

    @JsonIgnore
    public Boolean getBinningMergeEnable() {
        return binningMergeEnable;
    }

    public void setBinningMergeEnable(Boolean binningMergeEnable) {
        this.binningMergeEnable = binningMergeEnable;
    }

    public BinningMethod getBinningMethod() {
        return binningMethod;
    }

    public void setBinningMethod(BinningMethod binningMethod) {
        this.binningMethod = binningMethod;
    }

    public Double getSampleRate() {
        return sampleRate;
    }

    public void setSampleRate(Double sampleRate) {
        this.sampleRate = sampleRate;
    }

    public Boolean getSampleNegOnly() {
        return sampleNegOnly;
    }

    public void setSampleNegOnly(Boolean sampleNegOnly) {
        this.sampleNegOnly = sampleNegOnly;
    }


    public String getRawStatsCalculator() {
        return rawStatsCalculator;
    }

    public void setRawStatsCalculator(String rawStatsCalculator) {
        this.rawStatsCalculator = rawStatsCalculator;
    }

    public String getNumBinningCalculator() {
        return numBinningCalculator;
    }

    public void setNumBinningCalculator(String numBinningCalculator) {
        this.numBinningCalculator = numBinningCalculator;
    }

    public String getCatBinningCalculator() {
        return catBinningCalculator;
    }

    public void setCatBinningCalculator(String catBinningCalculator) {
        this.catBinningCalculator = catBinningCalculator;
    }

    public String getNumStatsCalculator() {
        return numStatsCalculator;
    }

    public void setNumStatsCalculator(String numStatsCalculator) {
        this.numStatsCalculator = numStatsCalculator;
    }

    public String getBinStatsCalculator() {
        return binStatsCalculator;
    }

    public void setBinStatsCalculator(String binStatsCalculator) {
        this.binStatsCalculator = binStatsCalculator;
    }
}
