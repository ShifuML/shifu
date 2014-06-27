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
package ml.shifu.core.container.obj;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * ModelNormalizeConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelNormalizeConf {

    private Double stdDevCutOff = Double.valueOf(4.0);
    private Double sampleRate = Double.valueOf(1.0);
    private Boolean sampleNegOnly = Boolean.FALSE;

    private String normalizer = "ml.core.core.di.builtin.WOENormalizer";

    public String getNormalizer() {
        return normalizer;
    }

    public void setNormalizer(String normalizer) {
        this.normalizer = normalizer;
    }
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

}
