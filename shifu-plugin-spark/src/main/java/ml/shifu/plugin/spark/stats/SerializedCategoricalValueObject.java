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
package ml.shifu.plugin.spark.stats;

import java.io.Serializable;
import java.util.Comparator;

import ml.shifu.core.container.CategoricalValueObject;

/**
 * Serialized version of ml.shifu.core.container.CategoricalValueObject 
 */
public class SerializedCategoricalValueObject
        implements Serializable {

    private static final long serialVersionUID = 1L;

    private Boolean isPositive;
    private String value;
    private Double weight;
    
    public SerializedCategoricalValueObject(String value, Double weight, Boolean isPositive) {
        this.isPositive= isPositive;
        this.weight= weight;
        this.value= value;
    }
    
    public SerializedCategoricalValueObject() {
        this.weight = 1.0;
    }
    
    public Boolean getIsPositive() {
        return isPositive;
    }

    public void setIsPositive(Boolean isPositive) {
        this.isPositive = isPositive;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

    public static class CategoricalValueObjectComparator implements Comparator<CategoricalValueObject> {

        public int compare(CategoricalValueObject a, CategoricalValueObject b) {
            int d = a.getValue().compareTo(b.getValue());
            if (d == 0) {
                return a.getIsPositive().compareTo(b.getIsPositive());
            } else {
                return d;
            }
        }
    }

}
