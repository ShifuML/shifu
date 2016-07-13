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
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "CorrelationWritable [columnIndex=" + columnIndex + ", sum=" + sum + ", sumSquare=" + sumSquare
                + ", count=" + count + ", xySum=" + Arrays.toString(xySum) + "]";
    }

}
