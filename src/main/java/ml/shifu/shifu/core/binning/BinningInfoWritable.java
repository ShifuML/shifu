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
package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.core.autotype.CountAndFrequentItemsWritable;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * {@link BinningInfoWritable} is to store column statistics collected from mapper and aggregated in reducer.
 */
public class BinningInfoWritable implements Writable {

    /**
     * A flag to not if no records in that mapper task, then in reducer, it will be ignored
     */
    private boolean isEmpty = false;

    private boolean isNumeric = true;

    private int columnNum;

    private List<Double> binBoundaries;

    private List<String> binCategories;

    private long[] binCountPos;

    private long[] binCountNeg;

    private double[] binWeightPos;

    private double[] binWeightNeg;

    private double min = Double.MAX_VALUE;

    private double max = Double.MIN_VALUE;

    private double sum = 0.0d;

    private double squaredSum = 0.0d;

    private double tripleSum = 0.0d;

    private double quarticSum = 0.0d;

    private long missingCount = 0L;

    private long totalCount = 0L;

    private double[] xMultiY = null;

    private CountAndFrequentItemsWritable cfiw = new CountAndFrequentItemsWritable();

    /**
     * @return the binBoundaries
     */
    public List<Double> getBinBoundaries() {
        return binBoundaries;
    }

    /**
     * @param binBoundaries
     *            the binBoundaries to set
     */
    public void setBinBoundaries(List<Double> binBoundaries) {
        this.binBoundaries = binBoundaries;
    }

    /**
     * @return the columnNum
     */
    public int getColumnNum() {
        return columnNum;
    }

    /**
     * @return the binCountPos
     */
    public long[] getBinCountPos() {
        return binCountPos;
    }

    /**
     * @return the binCountNeg
     */
    public long[] getBinCountNeg() {
        return binCountNeg;
    }

    /**
     * @return the binWeightPos
     */
    public double[] getBinWeightPos() {
        return binWeightPos;
    }

    /**
     * @return the binWeightNeg
     */
    public double[] getBinWeightNeg() {
        return binWeightNeg;
    }

    /**
     * @return the min
     */
    public double getMin() {
        return min;
    }

    /**
     * @return the max
     */
    public double getMax() {
        return max;
    }

    /**
     * @return the sum
     */
    public double getSum() {
        return sum;
    }

    /**
     * @return the squaredSum
     */
    public double getSquaredSum() {
        return squaredSum;
    }

    /**
     * @return the missingCount
     */
    public long getMissingCount() {
        return missingCount;
    }

    /**
     * @return the totalCount
     */
    public long getTotalCount() {
        return totalCount;
    }

    /**
     * @param columnNum
     *            the columnNum to set
     */
    public void setColumnNum(int columnNum) {
        this.columnNum = columnNum;
    }

    /**
     * @param binCountPos
     *            the binCountPos to set
     */
    public void setBinCountPos(long[] binCountPos) {
        this.binCountPos = binCountPos;
    }

    /**
     * @param binCountNeg
     *            the binCountNeg to set
     */
    public void setBinCountNeg(long[] binCountNeg) {
        this.binCountNeg = binCountNeg;
    }

    /**
     * @param binWeightPos
     *            the binWeightPos to set
     */
    public void setBinWeightPos(double[] binWeightPos) {
        this.binWeightPos = binWeightPos;
    }

    /**
     * @param binWeightNeg
     *            the binWeightNeg to set
     */
    public void setBinWeightNeg(double[] binWeightNeg) {
        this.binWeightNeg = binWeightNeg;
    }

    /**
     * @param min
     *            the min to set
     */
    public void setMin(double min) {
        this.min = min;
    }

    /**
     * @param max
     *            the max to set
     */
    public void setMax(double max) {
        this.max = max;
    }

    /**
     * @param sum
     *            the sum to set
     */
    public void setSum(double sum) {
        this.sum = sum;
    }

    /**
     * @param squaredSum
     *            the squaredSum to set
     */
    public void setSquaredSum(double squaredSum) {
        this.squaredSum = squaredSum;
    }

    /**
     * @param missingCount
     *            the missingCount to set
     */
    public void setMissingCount(long missingCount) {
        this.missingCount = missingCount;
    }

    /**
     * @param totalCount
     *            the totalCount to set
     */
    public void setTotalCount(long totalCount) {
        this.totalCount = totalCount;
    }

    /**
     * @return the binCategories
     */
    public List<String> getBinCategories() {
        return binCategories;
    }

    /**
     * @param binCategories
     *            the binCategories to set
     */
    public void setBinCategories(List<String> binCategories) {
        this.binCategories = binCategories;
    }

    /**
     * @return the isNumeric
     */
    public boolean isNumeric() {
        return isNumeric;
    }

    /**
     * @param isNumeric
     *            the isNumeric to set
     */
    public void setNumeric(boolean isNumeric) {
        this.isNumeric = isNumeric;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeBoolean(this.isNumeric);
        out.writeInt(this.columnNum);

        out.writeDouble(this.max);
        out.writeDouble(this.min);
        out.writeDouble(this.sum);
        out.writeDouble(this.squaredSum);
        out.writeDouble(this.tripleSum);
        out.writeDouble(this.quarticSum);

        out.writeLong(this.missingCount);
        out.writeLong(this.totalCount);

        out.writeInt(this.binCountPos.length);
        for(int i = 0; i < this.binCountPos.length; i++) {
            out.writeLong(this.binCountPos[i]);
        }

        out.writeInt(this.binCountNeg.length);
        for(int i = 0; i < this.binCountNeg.length; i++) {
            out.writeLong(this.binCountNeg[i]);
        }

        out.writeInt(this.binWeightPos.length);
        for(int i = 0; i < this.binWeightPos.length; i++) {
            out.writeDouble(this.binWeightPos[i]);
        }

        out.writeInt(this.binWeightNeg.length);
        for(int i = 0; i < this.binWeightNeg.length; i++) {
            out.writeDouble(this.binWeightNeg[i]);
        }

        // binBoundaries
        if(this.binBoundaries == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.binBoundaries.size());
            for(int i = 0; i < this.binBoundaries.size(); i++) {
                out.writeDouble(this.binBoundaries.get(i));
            }
        }

        if(this.xMultiY != null) {
            out.writeInt(this.xMultiY.length);
            for(double d: this.xMultiY) {
                out.writeDouble(d);
            }
        } else {
            out.writeInt(0);
        }

        // binCategories
        if(this.binCategories == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.binCategories.size());
            for(int i = 0; i < this.binCategories.size(); i++) {
                String bin = this.binCategories.get(i);
                byte[] bytes = bin.getBytes(Charset.forName("UTF-8"));
                out.writeInt(bytes.length);
                for(int j = 0; j < bytes.length; j++) {
                    out.writeByte(bytes[j]);
                }
            }
        }

        this.cfiw.write(out);
        out.writeBoolean(this.isEmpty);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.isNumeric = in.readBoolean();
        this.columnNum = in.readInt();
        this.max = in.readDouble();
        this.min = in.readDouble();
        this.sum = in.readDouble();
        this.squaredSum = in.readDouble();
        this.tripleSum = in.readDouble();
        this.quarticSum = in.readDouble();

        this.missingCount = in.readLong();
        this.totalCount = in.readLong();

        int size = in.readInt();
        this.binCountPos = new long[size];
        for(int i = 0; i < size; i++) {
            this.binCountPos[i] = in.readLong();
        }

        size = in.readInt();
        this.binCountNeg = new long[size];
        for(int i = 0; i < size; i++) {
            this.binCountNeg[i] = in.readLong();
        }

        size = in.readInt();
        this.binWeightPos = new double[size];
        for(int i = 0; i < size; i++) {
            this.binWeightPos[i] = in.readDouble();
        }

        size = in.readInt();
        this.binWeightNeg = new double[size];
        for(int i = 0; i < size; i++) {
            this.binWeightNeg[i] = in.readDouble();
        }

        // read binBoundaries
        size = in.readInt();
        this.binBoundaries = new ArrayList<Double>(size);
        for(int i = 0; i < size; i++) {
            this.binBoundaries.add(in.readDouble());
        }

        // read xMultiY
        int xMultiYSize = in.readInt();
        if(xMultiYSize != 0) {
            this.xMultiY = new double[xMultiYSize];
            for(int i = 0; i < xMultiYSize; i++) {
                this.xMultiY[i] = in.readDouble();
            }
        }

        // read binCategories
        size = in.readInt();
        this.binCategories = new ArrayList<String>(size);
        for(int i = 0; i < size; i++) {
            int bytesSize = in.readInt();
            byte[] bytes = new byte[bytesSize];
            for(int j = 0; j < bytesSize; j++) {
                bytes[j] = in.readByte();
            }
            this.binCategories.add(new String(bytes, Charset.forName("UTF-8")));
        }

        this.cfiw = new CountAndFrequentItemsWritable();
        this.cfiw.readFields(in);
        this.isEmpty = in.readBoolean();
    }

    /**
     * @return the tripleSum
     */
    public double getTripleSum() {
        return tripleSum;
    }

    /**
     * @param tripleSum
     *            the tripleSum to set
     */
    public void setTripleSum(double tripleSum) {
        this.tripleSum = tripleSum;
    }

    /**
     * @return the quarticSum
     */
    public double getQuarticSum() {
        return quarticSum;
    }

    /**
     * @param quarticSum
     *            the quarticSum to set
     */
    public void setQuarticSum(double quarticSum) {
        this.quarticSum = quarticSum;
    }

    /**
     * @return the xMultiY
     */
    public double[] getxMultiY() {
        return xMultiY;
    }

    /**
     * @param xMultiY
     *            the xMultiY to set
     */
    public void setxMultiY(double[] xMultiY) {
        this.xMultiY = xMultiY;
    }

    /**
     * @return the cfiw
     */
    public CountAndFrequentItemsWritable getCfiw() {
        return cfiw;
    }

    /**
     * @param cfiw
     *            the cfiw to set
     */
    public void setCfiw(CountAndFrequentItemsWritable cfiw) {
        this.cfiw = cfiw;
    }

    /**
     * @return the isEmpty
     */
    public boolean isEmpty() {
        return isEmpty;
    }

    /**
     * @param isEmpty
     *            the isEmpty to set
     */
    public void setEmpty(boolean isEmpty) {
        this.isEmpty = isEmpty;
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "BinningInfoWritable [isNumeric=" + isNumeric + ", columnNum=" + columnNum + ", binBoundaries="
                + binBoundaries + ", binCategories=" + binCategories + ", binCountPos=" + Arrays.toString(binCountPos)
                + ", binCountNeg=" + Arrays.toString(binCountNeg) + ", binWeightPos=" + Arrays.toString(binWeightPos)
                + ", binWeightNeg=" + Arrays.toString(binWeightNeg) + ", min=" + min + ", max=" + max + ", sum=" + sum
                + ", squaredSum=" + squaredSum + ", tripleSum=" + tripleSum + ", quarticSum=" + quarticSum
                + ", missingCount=" + missingCount + ", totalCount=" + totalCount + "]";
    }

}
