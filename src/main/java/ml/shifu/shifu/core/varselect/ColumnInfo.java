/*
 * Copyright [2012-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.varselect;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * Column info for sensitivity analysis.
 * 
 * <p>
 * Includes info for mean, RMS and variance computation.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class ColumnInfo implements Writable {

    private double sumScoreDiff;

    private long count;

    private double sumSquareScoreDiff;

    /**
     * @return the sumScoreDiff
     */
    public double getSumScoreDiff() {
        return sumScoreDiff;
    }

    /**
     * @param sumScoreDiff
     *            the sumScoreDiff to set
     */
    public void setSumScoreDiff(double sumScoreDiff) {
        this.sumScoreDiff = sumScoreDiff;
    }

    /**
     * @return the count
     */
    public long getCount() {
        return count;
    }

    /**
     * @param count
     *            the count to set
     */
    public void setCount(long count) {
        this.count = count;
    }

    /**
     * @return the sumSquareScoreDiff
     */
    public double getSumSquareScoreDiff() {
        return sumSquareScoreDiff;
    }

    /**
     * @param sumSquareScoreDiff
     *            the sumSquareScoreDiff to set
     */
    public void setSumSquareScoreDiff(double sumSquareScoreDiff) {
        this.sumSquareScoreDiff = sumSquareScoreDiff;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeDouble(sumScoreDiff);
        out.writeLong(count);
        out.writeDouble(sumSquareScoreDiff);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.sumScoreDiff = in.readDouble();
        this.count = in.readLong();
        this.sumSquareScoreDiff = in.readDouble();
    }

    /* (non-Javadoc)
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "ColumnInfo [sumScoreDiff=" + sumScoreDiff + ", count=" + count + ", sumSquareScoreDiff="
                + sumSquareScoreDiff + "]";
    }

}
