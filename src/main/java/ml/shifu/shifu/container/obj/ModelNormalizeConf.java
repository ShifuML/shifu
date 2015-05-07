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

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * ModelNormalizeConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelNormalizeConf {

    @JsonDeserialize(using = NormTypeDeserializer.class)
    public static enum NormType {
        ZSCALE, WOE;
    }
    
    @JsonDeserialize(using = MissValueFillTypeDeserializer.class)
    public static enum MissValueFillType {
        MEAN, COUNTWOE, WEIGHTEDWOE, ZERO;
    }
    
    private Double stdDevCutOff = Double.valueOf(4.0);
    private Double sampleRate = Double.valueOf(1.0);
    private Boolean sampleNegOnly = Boolean.FALSE;
    private NormType normType = NormType.ZSCALE;
    private MissValueFillType missValueFillType = MissValueFillType.MEAN;
    private Boolean isWeightNorm = Boolean.FALSE;

    // move to RawSourceData
    // private String weightAmplifier;
    // private List<WeightAmplifier> weightAmplifier;

    public Double getStdDevCutOff() {
        return stdDevCutOff;
    }

    public void setStdDevCutOff(Double stdDevCutOff) {
        this.stdDevCutOff = stdDevCutOff;
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

    /**
     * @return the normType
     */
    public NormType getNormType() {
        return normType;
    }

    /**
     * @param normType the normType to set
     */
    public void setNormType(NormType normType) {
        this.normType = normType;
    }

    /**
     * @return the fillType
     */
    public MissValueFillType getMissValueFillType() {
        return missValueFillType;
    }

    /**
     * @param fillType the fillType to set
     */
    public void setMissValueFillType(MissValueFillType fillType) {
        this.missValueFillType = fillType;
    }

    /**
     * @return the isWeightNorm
     */
    public Boolean getIsWeightNorm() {
        return isWeightNorm;
    }

    /**
     * @param isWeightNorm the isWeightNorm to set
     */
    public void setIsWeightNorm(Boolean isWeightNorm) {
        this.isWeightNorm = isWeightNorm;
    }

}
