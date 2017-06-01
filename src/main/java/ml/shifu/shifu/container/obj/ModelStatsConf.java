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
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * {@link ModelStatsConf} is 'stats' part configuration in ModelConfig.json
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelStatsConf {

    /**
     * Binning strategy used in stats step.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    @JsonDeserialize(using = BinningMethodDeserializer.class)
    public static enum BinningMethod {
        EqualNegtive, EqualInterval, EqualPositive, EqualTotal, WeightEqualNegative, WeightEqualInterval, WeightEqualPositive, WeightEqualTotal
    }

    /**
     * Binning algorithm on how to scale binning in 10k features well.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    public static enum BinningAlgorithm {
        Native, // sorting way
        SPDT, // paper reference: www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
        SPDTI, // paper reference: www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf, improvement for last
               // binning updating step
        MunroPat, // paper reference: www.cs.ucsb.edu/~suri/cs290/MunroPat.pdf
        MunroPatI, // paper reference: www.cs.ucsb.edu/~suri/cs290/MunroPat.pdf, improvement for last
                   // binning updating step
        DynamicBinning
    }

    /**
     * Max num bin per each numerical column.
     */
    private Integer maxNumBin = 10;

    /**
     * Max num bin for each categorical column
     */
    private Integer cateMaxNumBin = 0;

    /**
     * Binning method used in stats. By default is EqualPositive.
     */
    private BinningMethod binningMethod = BinningMethod.EqualPositive;

    /**
     * Sampling rate in stats step. Sometimes is binning algorithm cannot be scaled well or slow. Try using smaller
     * sampleRate will accelerate stats.
     */
    private Double sampleRate = Double.valueOf(1.0);

    /**
     * If only sample negative records or not, positive records in most cases is less than negative. By only sampling
     * negative can balance data.
     */
    private Boolean sampleNegOnly = Boolean.FALSE;

    // don't open those options to user, this only works in some binning algorithm
    private Double numericalValueThreshold = Double.MAX_VALUE;
    private Boolean binningAutoTypeEnable = Boolean.FALSE;
    private Integer binningAutoTypeThreshold = 5;
    private Boolean binningMergeEnable = Boolean.TRUE;

    /**
     * Binning algorithm used to do binning. SPDTI is the best algorithm in terms of scalability.
     */
    private BinningAlgorithm binningAlgorithm = BinningAlgorithm.SPDTI;

    /**
     * PSI feature enabled if not empty. In stats, PSI value will be computed.
     */
    private String psiColumnName = "";

    public Integer getMaxNumBin() {
        return maxNumBin;
    }

    public void setMaxNumBin(Integer maxNumBin) {
        this.maxNumBin = maxNumBin;
    }

    public Integer getCateMaxNumBin() {
        return cateMaxNumBin;
    }

    public void setCateMaxNumBin(Integer cateMaxNumBin) {
        this.cateMaxNumBin = cateMaxNumBin;
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

    public BinningAlgorithm getBinningAlgorithm() {
        return binningAlgorithm;
    }

    public void setBinningAlgorithm(BinningAlgorithm binningAlgorithm) {
        this.binningAlgorithm = binningAlgorithm;
    }

    public String getPsiColumnName() {
        return psiColumnName;
    }

    public void setPsiColumnName(String psiColumnName) {
        this.psiColumnName = psiColumnName;
    }

    @Override
    public ModelStatsConf clone() {
        ModelStatsConf other = new ModelStatsConf();
        other.setBinningAlgorithm(binningAlgorithm);
        other.setBinningAutoTypeEnable(binningAutoTypeEnable);
        other.setBinningAutoTypeThreshold(binningAutoTypeThreshold);
        other.setBinningMergeEnable(binningMergeEnable);
        other.setBinningMethod(binningMethod);
        other.setMaxNumBin(maxNumBin);
        other.setNumericalValueThreshold(numericalValueThreshold);
        other.setPsiColumnName(psiColumnName);
        other.setSampleNegOnly(sampleNegOnly);
        other.setSampleRate(sampleRate);
        return other;
    }
}
