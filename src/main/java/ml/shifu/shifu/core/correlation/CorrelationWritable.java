/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.correlation;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;

/**
 * {@link CorrelationWritable} is used to store infomation which is used to compute pearson correlation between two
 * variables.
 * 
 * <p>
 * Within pearson correlation, sum, sum of squared, count and sum of x*y are all computed and set it into this writable
 * instance.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class CorrelationWritable implements Writable {

    private int columnIndex;

    private double sum;

    private double sumSquare;

    private double count;

    private double[] xySum;

    private double[] xxSum;

    private double[] yySum;

    private double[] adjustCount;

    private double[] adjustSumX;

    private double[] adjustSumY;

    /**
     * @return the columnIndex
     */
    public int getColumnIndex() {
        return columnIndex;
    }

    /**
     * @return the sum
     */
    public double getSum() {
        return sum;
    }

    /**
     * @return the sumSquare
     */
    public double getSumSquare() {
        return sumSquare;
    }

    /**
     * @return the count
     */
    public double getCount() {
        return count;
    }

    /**
     * @return the xySum
     */
    public double[] getXySum() {
        return xySum;
    }

    /**
     * @return the xxSum
     */
    public double[] getXxSum() {
        return xxSum;
    }

    /**
     * @return the yySum
     */
    public double[] getYySum() {
        return yySum;
    }

    /**
     * @return the adjustCount
     */
    public double[] getAdjustCount() {
        return adjustCount;
    }

    /**
     * @return the adjustSumX
     */
    public double[] getAdjustSumX() {
        return adjustSumX;
    }

    /**
     * @return the adjustSumY
     */
    public double[] getAdjustSumY() {
        return adjustSumY;
    }

    /**
     * @param columnIndex
     *            the columnIndex to set
     */
    public void setColumnIndex(int columnIndex) {
        this.columnIndex = columnIndex;
    }

    /**
     * @param sum
     *            the sum to set
     */
    public void setSum(double sum) {
        this.sum = sum;
    }

    /**
     * @param sumSquare
     *            the sumSquare to set
     */
    public void setSumSquare(double sumSquare) {
        this.sumSquare = sumSquare;
    }

    /**
     * @param count
     *            the count to set
     */
    public void setCount(double count) {
        this.count = count;
    }

    /**
     * @param xySum
     *            the xySum to set
     */
    public void setXySum(double[] xySum) {
        this.xySum = xySum;
    }

    /**
     * @param xxSum
     *            the xxSum to set
     */
    public void setXxSum(double[] xxSum) {
        this.xxSum = xxSum;
    }

    /**
     * @param yySum
     *            the yySum to set
     */
    public void setYySum(double[] yySum) {
        this.yySum = yySum;
    }

    /**
     * @param adjustCount
     *            the adjustCount to set
     */
    public void setAdjustCount(double[] adjustCount) {
        this.adjustCount = adjustCount;
    }

    /**
     * @param adjustSumX
     *            the adjustSumX to set
     */
    public void setAdjustSumX(double[] adjustSumX) {
        this.adjustSumX = adjustSumX;
    }

    /**
     * @param adjustSumY
     *            the adjustSumY to set
     */
    public void setAdjustSumY(double[] adjustSumY) {
        this.adjustSumY = adjustSumY;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.columnIndex);
        out.writeDouble(this.sum);
        out.writeDouble(this.sumSquare);
        out.writeDouble(this.count);
        if(this.xySum == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.xySum.length);
            for(double doub: this.xySum) {
                out.writeDouble(doub);
            }
        }

        if(this.xxSum == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.xxSum.length);
            for(double doub: this.xxSum) {
                out.writeDouble(doub);
            }
        }

        if(this.yySum == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.yySum.length);
            for(double doub: this.yySum) {
                out.writeDouble(doub);
            }
        }

        if(this.adjustCount == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.adjustCount.length);
            for(double doub: this.adjustCount) {
                out.writeDouble(doub);
            }
        }

        if(this.adjustSumX == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.adjustSumX.length);
            for(double doub: this.adjustSumX) {
                out.writeDouble(doub);
            }
        }

        if(this.adjustSumY == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.adjustSumY.length);
            for(double doub: this.adjustSumY) {
                out.writeDouble(doub);
            }
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.columnIndex = in.readInt();
        this.sum = in.readDouble();
        this.sumSquare = in.readDouble();
        this.count = in.readDouble();

        int length = in.readInt();
        this.xySum = new double[length];
        for(int i = 0; i < length; i++) {
            this.xySum[i] = in.readDouble();
        }

        length = in.readInt();
        this.xxSum = new double[length];
        for(int i = 0; i < length; i++) {
            this.xxSum[i] = in.readDouble();
        }

        length = in.readInt();
        this.yySum = new double[length];
        for(int i = 0; i < length; i++) {
            this.yySum[i] = in.readDouble();
        }

        length = in.readInt();
        this.adjustCount = new double[length];
        for(int i = 0; i < length; i++) {
            this.adjustCount[i] = in.readDouble();
        }

        length = in.readInt();
        this.adjustSumX = new double[length];
        for(int i = 0; i < length; i++) {
            this.adjustSumX[i] = in.readDouble();
        }

        length = in.readInt();
        this.adjustSumY = new double[length];
        for(int i = 0; i < length; i++) {
            this.adjustSumY[i] = in.readDouble();
        }
    }

    public CorrelationWritable combine(CorrelationWritable from) {
        this.sum += from.sum;
        this.sumSquare += from.sumSquare;
        this.count += from.count;

        for(int i = 0; i < xySum.length; i++) {
            this.xySum[i] += from.xySum[i];
        }

        for(int i = 0; i < xxSum.length; i++) {
            this.xxSum[i] += from.xxSum[i];
        }

        for(int i = 0; i < yySum.length; i++) {
            this.yySum[i] += from.yySum[i];
        }

        for(int i = 0; i < adjustCount.length; i++) {
            this.adjustCount[i] += from.adjustCount[i];
        }

        for(int i = 0; i < adjustSumX.length; i++) {
            this.adjustSumX[i] += from.adjustSumX[i];
        }

        for(int i = 0; i < adjustSumY.length; i++) {
            this.adjustSumY[i] += from.adjustSumY[i];
        }
        return this;
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "CorrelationWritable [columnIndex=" + columnIndex + ", sum=" + sum + ", sumSquare=" + sumSquare
                + ", count=" + count + ", xySum=" + Arrays.toString(xySum) + ", xxSum=" + Arrays.toString(xxSum)
                + ", yySum=" + Arrays.toString(yySum) + ", adjustCount=" + Arrays.toString(adjustCount)
                + ", adjustSumX=" + Arrays.toString(adjustSumX) + ", adjustSumY=" + Arrays.toString(adjustSumY) + "]";
    }

}
