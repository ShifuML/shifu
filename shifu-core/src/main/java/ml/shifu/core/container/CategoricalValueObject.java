/**
 * Copyright [2012-2013] eBay Software Foundation
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
package ml.shifu.core.container;

import java.util.Comparator;

/**
 * Categorical Value Object
 */
public class CategoricalValueObject {

    private Boolean isPositive;
    private String value;
    private Double weight;
    public CategoricalValueObject() {
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
            int d = a.value.compareTo(b.value);
            if (d == 0) {
                return a.getIsPositive().compareTo(b.getIsPositive());
            } else {
                return d;
            }
        }
    }

}
