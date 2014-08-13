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
package ml.shifu.core.container;

import java.util.Comparator;

/**
 * Binning object, data with its type
 */
public class BinningObject {

    private final DataType type;

    private Double nData;
    private String cData;
    private Double score;
    private String tag;
    public BinningObject(DataType type) {
        this.type = type;
    }

    // Getters
    public DataType getType() {
        return this.type;
    }

    public Double getNumericalData() {
        return this.nData;
    }

    // Setters
    public void setNumericalData(Double data) {
        this.nData = data;
    }

    public String getCategoricalData() {
        if (this.type.equals(DataType.Numerical)) {
            throw new RuntimeException(
                    "Cannot get categorical data from a numerical variable.");
        }
        return this.cData;
    }

    public void setCategoricalData(String data) {
        if (this.type.equals(DataType.Numerical)) {
            throw new RuntimeException(
                    "Cannot set categorical data to a numerical variable.");
        }
        this.cData = data;
    }

    public Double getScore() {
        return this.score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    // Others
    public String toString() {
        if (this.type.equals(DataType.Categorical)) {
            return "(" + this.cData + ", " + this.tag + ", " + this.score + ")";
        } else {
            return "(" + this.nData + ", " + this.tag + ", " + this.score + ")";
        }
    }

    public static enum DataType {
        Numerical, Categorical
    }

    // Comparator
    public static class VariableObjectComparator implements
            Comparator<BinningObject> {
        public int compare(BinningObject a, BinningObject b) {
            if (a.type.equals(DataType.Categorical)
                    && b.type.equals(DataType.Categorical)) {
                int d = a.cData.compareTo(b.cData);
                if (d == 0) {
                    return a.tag.compareTo(b.tag);
                } else {
                    return d;
                }
            } else if (a.type.equals(DataType.Numerical)
                    && b.type.equals(DataType.Numerical)) {
                int d = a.nData.compareTo(b.nData);
                if (d == 0) {
                    return a.tag.compareTo(b.tag);
                } else {
                    return d;
                }
            } else {
                throw new RuntimeException("Data should be in the same type");
            }
        }
    }
}
