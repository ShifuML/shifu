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
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * {@link ModelNormalizeConf} is 'nomalize' part configuration in ModelConfig.json
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelNormalizeConf {

    /**
     * Normalization type including ZSCALE, WOE, WEIGHT_WOE, HYBRID, WEIGHT_HYBRID.
     */
    @JsonDeserialize(using = NormTypeDeserializer.class)
    public static enum NormType {
        OLD_ZSCORE, OLD_ZSCALE, // the same one for user friendly
        ZSCORE, ZSCALE, // the same one for user friendly
        WOE, WEIGHT_WOE, HYBRID, WEIGHT_HYBRID, 
        WOE_ZSCORE, WOE_ZSCALE, 
        WEIGHT_WOE_ZSCORE, WEIGHT_WOE_ZSCALE,
        ZSCALE_ONEHOT,
        DISCRETE_ZSCORE, DISCRETE_ZSCALE // for numerical feature, use low bondwary in each bin, the first bin use min 
                        // value, missing value use raw mean value, then do zscale by raw mean and raw std-dev;
                        // for categorical feature, pos rate is used.
        ;

        public boolean isWoe() {
            return this == WOE || this == WEIGHT_WOE || this == WOE_ZSCORE || this == WOE_ZSCALE
                    || this == WEIGHT_WOE_ZSCORE || this == WEIGHT_WOE_ZSCORE;
        }
    }

    /**
     * Different correlation computing methods.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    @JsonDeserialize(using = CorrelationDeserializer.class)
    public static enum Correlation {
        None, Pearson, NormPearson // Spearman mode isn't implemented as need sort all variables
    }

    /**
     * STDDev cutoff threshold, if over this value after zscore, such value will be cutoff to current value or negative
     * of this value.
     */
    private Double stdDevCutOff = Double.valueOf(4.0);

    /**
     * If do sampling in norm step, training will be impacted by sampling because norm output is train input
     */
    private Double sampleRate = Double.valueOf(1.0);

    /**
     * If only sample negative with sampleRate enabled
     */
    private Boolean sampleNegOnly = Boolean.FALSE;

    /**
     * Different norm type
     */
    private NormType normType = NormType.ZSCALE;

    /**
     * If norm output is parquet format, if parquet format and only part of features are selected, in training, only
     * selected columns are read. So far Parquet format only supports NN algorithm.
     */
    private Boolean isParquet = Boolean.FALSE;

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
     * @param normType
     *            the normType to set
     */
    public void setNormType(NormType normType) {
        this.normType = normType;
    }

    /**
     * @return the isParquet
     */
    @JsonIgnore
    public Boolean getIsParquet() {
        return isParquet;
    }

    /**
     * @param isParquet
     *            the isParquet to set
     */
    @JsonProperty
    public void setIsParquet(Boolean isParquet) {
        this.isParquet = isParquet;
    }

    @Override
    public ModelNormalizeConf clone() {
        ModelNormalizeConf other = new ModelNormalizeConf();
        other.setNormType(normType);
        other.setSampleRate(sampleRate);
        other.setSampleNegOnly(sampleNegOnly);
        other.setStdDevCutOff(stdDevCutOff);
        other.setIsParquet(isParquet);
        // other.setCorrelation(correlation);
        return other;
    }
}
