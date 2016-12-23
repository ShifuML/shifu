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
package ml.shifu.shifu.container;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * WeighterExpressin class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class WeightAmplifier {

    private Double targetWeight = 1.0d;
    private String targetExpression;

    public WeightAmplifier() {
        super();
    }

    public WeightAmplifier(Double targetWeight, String targetExpression) {
        super();
        this.targetWeight = targetWeight;
        this.targetExpression = targetExpression;
    }

    public Double getTargetWeight() {
        return targetWeight;
    }

    public void setTargetWeight(Double targetWeight) {
        this.targetWeight = targetWeight;
    }

    public String getTargetExpression() {
        return targetExpression;
    }

    public void setTargetExpression(String targetExpression) {
        this.targetExpression = targetExpression;
    }

}
