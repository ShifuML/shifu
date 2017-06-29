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
import java.util.Set;

import ml.shifu.guagua.io.Bytable;

/**
 * Split for Both continuous and categorical features.
 * 
 * <p>
 * For continuous feature, only a double threshold can be used to split a variable into two splits. While for
 * categorical features, we only store left node category list, check if in left category list to determine which split.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Split implements Bytable {

    /**
     * Column number in ColumnConfig.json
     */
    private int columnNum;

    /**
     * CONTINUOUS or CATEGORICAL, should not be null
     */
    private byte featureType;

    /**
     * For CONTINUOUS feature, this should be valid value to split feature
     */
    private double threshold;

    /**
     * For categorical feature, if isLeft = true, {@link #leftOrRightCategories} stores left categories. If false,
     * {@link #leftOrRightCategories} stores right categories.
     */
    private boolean isLeft = true;;

    /**
     * Indexes of left categories or right categories, list of categories will be saved in model files or in
     * TreeModel as short indexes to save space, short is safe so far as max bin size is limit to Short.MAX_VALUE.
     */
    private Set<Short> leftOrRightCategories;

    public final static byte CONTINUOUS = 1;
    public final static byte CATEGORICAL = 2;

    public Split() {
    }

    public Split(int columnNum, byte featureType, double threshold, boolean isLeft, Set<Short> leftOrRightCategories) {
        this.columnNum = columnNum;
        this.featureType = featureType;
        this.threshold = threshold;
        this.isLeft = isLeft;
        this.leftOrRightCategories = leftOrRightCategories;
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
    public byte getFeatureType() {
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
    public Set<Short> getLeftOrRightCategories() {
        return leftOrRightCategories;
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
    public void setFeatureType(byte featureType) {
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
    public void setLeftOrRightCategories(Set<Short> leftCategories) {
        this.leftOrRightCategories = leftCategories;
    }

    /**
     * @return the isLeft
     */
    public boolean isLeft() {
        return isLeft;
    }

    /**
     * @param isLeft
     *            the isLeft to set
     */
    public void setLeft(boolean isLeft) {
        this.isLeft = isLeft;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.columnNum);
        // use byte type to save space, should not be null
        out.writeByte(this.featureType);

        switch(this.featureType) {
            case CATEGORICAL:
                out.writeBoolean(this.isLeft);
                if(leftOrRightCategories == null) {
                    out.writeBoolean(true);
                } else {
                    out.writeBoolean(false);
                    if(leftOrRightCategories instanceof Bytable) {
                        ((Bytable) leftOrRightCategories).write(out);
                    }
                }
                break;
            case CONTINUOUS:
                out.writeDouble(this.threshold);
                break;
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.columnNum = in.readInt();
        this.featureType = in.readByte();

        switch(this.featureType) {
            case CATEGORICAL:
                this.isLeft = in.readBoolean();
                boolean isNull = in.readBoolean();
                if(isNull) {
                    leftOrRightCategories = null;
                } else {
                    leftOrRightCategories = new SimpleBitSet<Short>();
                    ((Bytable) leftOrRightCategories).readFields(in);
                }
                break;
            case CONTINUOUS:
                this.threshold = in.readDouble();
                break;
        }
    }

    @Override
    public String toString() {
        return "Split [featureIndex=" + columnNum + ", featureType=" + featureType + ", threshold=" + threshold
                + ", leftCategories=" + leftOrRightCategories + "]";
    }

}
