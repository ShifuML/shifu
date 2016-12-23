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

import java.io.Serializable;
import java.util.Comparator;

import ml.shifu.shifu.core.Binning.BinningDataType;

/**
 * data input object
 */
public class ValueObject {

    private String raw;
    private String tag;
    private Double value;
    private Double weight;

    public ValueObject() {
        this.weight = 1.0;
    }

    public static class ValueObjectComparator implements Comparator<ValueObject>, Serializable {

        private static final long serialVersionUID = -6290803062854530962L;

        private BinningDataType type;

        public ValueObjectComparator(BinningDataType type) {
            this.type = type;
        }

        public int compare(ValueObject a, ValueObject b) {
            if(type.equals(BinningDataType.Categorical)) {
                int d = a.raw.compareTo(b.raw);
                if(d == 0) {
                    return a.tag.compareTo(b.tag);
                } else {
                    return d;
                }
            } else {
                int d = a.value.compareTo(b.value);
                if(d == 0) {
                    return a.tag.compareTo(b.tag);
                } else {
                    return d;
                }
            }
        }
    }

    public static class WeightValueObjectComparator implements Comparator<ValueObject>, Serializable {

        private static final long serialVersionUID = -2312088241656723511L;
        
        private BinningDataType type;

        public WeightValueObjectComparator(BinningDataType type) {
            this.type = type;
        }

        @Override
        public int compare(ValueObject a, ValueObject b) {
            if(type.equals(BinningDataType.Categorical)) {
                int d = a.raw.compareTo(b.raw);
                if(d == 0) {
                    return a.tag.compareTo(b.tag);
                } else {
                    return d;
                }
            } else {
                Double weightA = a.value * a.weight;
                Double weightB = b.value * b.weight;
                int d = weightA.compareTo(weightB);
                if(d == 0) {
                    return a.tag.compareTo(b.tag);
                } else {
                    return d;
                }
            }
        }

    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    public String getRaw() {
        return raw;
    }

    public void setRaw(String raw) {
        this.raw = raw;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

}
