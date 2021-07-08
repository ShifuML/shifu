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

    private double[] binCountWoe;

    private double[] binWeightedWoe;

    private double min = Double.MAX_VALUE;

    private double max = Double.MIN_VALUE;

    private double sum = 0.0d;

    private double squaredSum = 0.0d;

    private double tripleSum = 0.0d;

    private double quarticSum = 0.0d;

    private long missingCount = 0L;

    private long totalCount = 0L;

    private double[] xMultiY = null;

    private int hashSeed = -1;

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
     * @return binCountWoe
     */
    public double[] getBinCountWoe() {
        return binCountWoe;
    }

    /**
     * @param binCountWoe
     *      the binCountWoe to set
     */
    public void setBinCountWoe(double[] binCountWoe) {
        this.binCountWoe = binCountWoe;
    }

    /**
     * @return binWeightedWoe
     */
    public double[] getBinWeightedWoe() {
        return binWeightedWoe;
    }

    /**
     * @param binWeightedWoe
     *      the binWeightedWoe to set
     */
    public void setBinWeightedWoe(double[] binWeightedWoe) {
        this.binWeightedWoe = binWeightedWoe;
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

        writeLongArray(out, this.binCountPos);
        writeLongArray(out, this.binCountNeg);

        writeDoubleArray(out, this.binWeightPos);
        writeDoubleArray(out, this.binWeightNeg);

        writeDoubleArray(out, this.binCountWoe);
        writeDoubleArray(out, this.binWeightedWoe);

        // binBoundaries
        writeDoubleList(out, this.binBoundaries);

        // xMultiY
        writeDoubleArray(out, this.xMultiY);

        // binCategories
        writeStringList(out, this.binCategories);

        this.cfiw.write(out);
        out.writeBoolean(this.isEmpty);
        out.writeInt(this.hashSeed);
    }

    private void writeLongArray(DataOutput out, long[] outputArray) throws IOException {
        if (outputArray == null) {
            out.writeInt(0);
        } else {
            out.writeInt(outputArray.length);
            for(int i = 0; i < outputArray.length; i++) {
                out.writeLong(outputArray[i]);
            }
        }
    }

    private void writeDoubleArray(DataOutput out, double[] outputArray) throws IOException {
        if (outputArray == null) {
            out.writeInt(0);
        } else {
            out.writeInt(outputArray.length);
            for(int i = 0; i < outputArray.length; i++) {
                out.writeDouble(outputArray[i]);
            }
        }
    }

    private void writeDoubleList(DataOutput out, List<Double> outputList) throws IOException {
        if (outputList == null) {
            out.writeInt(0);
        } else {
            out.writeInt(outputList.size());
            for(int i = 0; i < outputList.size(); i++) {
                out.writeDouble(outputList.get(i));
            }
        }
    }

    private void writeStringList(DataOutput out, List<String> outputList) throws IOException {
        if (outputList == null) {
            out.writeInt(0);
        } else {
            out.writeInt(outputList.size());
            for(int i = 0; i < outputList.size(); i++) {
                String bin = outputList.get(i);
                byte[] bytes = bin.getBytes(Charset.forName("UTF-8"));
                out.writeInt(bytes.length);
                for(int j = 0; j < bytes.length; j++) {
                    out.writeByte(bytes[j]);
                }
            }
        }
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

        this.binCountPos = readLongArray(in);
        this.binCountNeg = readLongArray(in);

        this.binWeightPos = readDoubleArray(in);
        this.binWeightNeg = readDoubleArray(in);

        this.binCountWoe = readDoubleArray(in);
        this.binWeightedWoe = readDoubleArray(in);

        // read binBoundaries
        this.binBoundaries = readDoubleList(in);

        // read xMultiY
        this.xMultiY = readDoubleArray(in);

        // read binCategories
        this.binCategories = readStringList(in);

        this.cfiw = new CountAndFrequentItemsWritable();
        this.cfiw.readFields(in);
        this.isEmpty = in.readBoolean();
        this.hashSeed = in.readInt();
    }

    private long[] readLongArray(DataInput in) throws IOException {
        int size = in.readInt();
        long[] valueArr = new long[size];
        for(int i = 0; i < size; i++) {
            valueArr[i] = in.readLong();
        }
        return valueArr;
    }

    private double[] readDoubleArray(DataInput in) throws IOException {
        int size = in.readInt();
        double[] valueArr = new double[size];
        for(int i = 0; i < size; i++) {
            valueArr[i] = in.readDouble();
        }
        return valueArr;
    }

    private List<Double> readDoubleList(DataInput in) throws IOException {
        int size = in.readInt();
        List<Double> valueList = new ArrayList<>();
        for(int i = 0; i < size; i++) {
            valueList.add(in.readDouble());
        }
        return valueList;
    }

    private List<String> readStringList(DataInput in) throws IOException {
        int size = in.readInt();
        List<String> valueList = new ArrayList<String>(size);
        for(int i = 0; i < size; i++) {
            int bytesSize = in.readInt();
            byte[] bytes = new byte[bytesSize];
            for(int j = 0; j < bytesSize; j++) {
                bytes[j] = in.readByte();
            }
            valueList.add(new String(bytes, Charset.forName("UTF-8")));
        }
        return valueList;
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

    /**
     * @return the hashSeed
     */
    public int getHashSeed() {
        return hashSeed;
    }

    /**
     * @param hashSeed
     *            the hashSeed to set
     */
    public void setHashSeed(int hashSeed) {
        this.hashSeed = hashSeed;
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
                + ", binWeightNeg=" + Arrays.toString(binWeightNeg) + ", binCountWoe=" + Arrays.toString(binCountWoe)
                + ", binWeightedWoe=" + Arrays.toString(binWeightedWoe)  + ", min=" + min + ", max=" + max + ", sum=" + sum
                + ", squaredSum=" + squaredSum + ", tripleSum=" + tripleSum + ", quarticSum=" + quarticSum
                + ", missingCount=" + missingCount + ", totalCount=" + totalCount + "]";
    }

}
