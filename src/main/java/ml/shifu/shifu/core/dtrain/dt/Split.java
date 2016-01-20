/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import ml.shifu.guagua.io.Bytable;

/**
 * Split for Both continuous and categorical features.
 * 
 * <p>
 * For continuous feature, only a double threshold can be used to split a variable into two splits. While for
 * categorical features, we only store left node category list, check if in left category list to determince which
 * split.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Split implements Bytable {

    private int columnNum;

    private FeatureType featureType;

    private double threshold;

    private Set<String> leftCategories;

    public Split() {
    }

    public Split(int featureIndex) {
        this.columnNum = featureIndex;
    }

    public Split(int columnNum, FeatureType featureType, double threshold, Set<String> leftCategories) {
        this.columnNum = columnNum;
        this.featureType = featureType;
        this.threshold = threshold;
        this.leftCategories = leftCategories;
    }

    /**
     * @return the featureIndex
     */
    public int getColumnNum() {
        return columnNum;
    }

    /**
     * @return the featureType
     */
    public FeatureType getFeatureType() {
        return featureType;
    }

    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @return the leftCategories
     */
    public Set<String> getLeftCategories() {
        return leftCategories;
    }

    /**
     * @param columnNum
     *            the columnNum to set
     */
    public void setColumnNum(int columnNum) {
        this.columnNum = columnNum;
    }

    /**
     * @param featureType
     *            the featureType to set
     */
    public void setFeatureType(FeatureType featureType) {
        this.featureType = featureType;
    }

    /**
     * @param threshold
     *            the threshold to set
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * @param leftCategories
     *            the leftCategories to set
     */
    public void setLeftCategories(Set<String> leftCategories) {
        this.leftCategories = leftCategories;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.columnNum);
        out.writeDouble(this.threshold);
        if(featureType == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeUTF(this.featureType.toString());
        }

        if(leftCategories == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.leftCategories.size());
            for(String category: this.leftCategories) {
                out.writeUTF(category);
            }
        }

    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.columnNum = in.readInt();
        this.threshold = in.readDouble();

        if(in.readBoolean()) {
            this.featureType = FeatureType.of(in.readUTF());
        }

        int len = in.readInt();
        this.leftCategories = new HashSet<String>();
        for(int i = 0; i < len; i++) {
            this.leftCategories.add(in.readUTF());
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "Split [featureIndex=" + columnNum + ", featureType=" + featureType + ", threshold=" + threshold
                + ", leftCategories=" + leftCategories + "]";
    }

}
