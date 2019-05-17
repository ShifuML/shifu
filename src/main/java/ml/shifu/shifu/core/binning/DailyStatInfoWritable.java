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

import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.autotype.CountAndFrequentItemsWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

/**
 * {@link DailyStatInfoWritable} is to store column statistics collected from mapper and aggregated in reducer.
 */
public class DailyStatInfoWritable implements Writable {

    /**
     * A flag to not if no records in that mapper task, then in reducer, it will be ignored
     */
    private boolean isEmpty = false;

    //variable, date, VariableStatInfo
    private Map<String, VariableStatInfo> variableDailyStatInfo = new HashMap<>();

    public Map<String, VariableStatInfo> getVariableDailyStatInfo() {
        return variableDailyStatInfo;
    }

    public void setVariableDailyStatInfo(Map<String, VariableStatInfo> variableDailyStatInfo) {
        this.variableDailyStatInfo = variableDailyStatInfo;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(variableDailyStatInfo.size());
        for (Map.Entry<String, VariableStatInfo> entry : variableDailyStatInfo.entrySet()) {
            out.writeInt(entry.getKey().length());
            for (byte b : entry.getKey().getBytes(Charset.forName("UTF-8"))){
                out.writeByte(b);
            }
            out.writeInt(entry.getValue().getColumnConfigIndex());

            out.writeDouble(entry.getValue().max);
            out.writeDouble(entry.getValue().min);
            out.writeDouble(entry.getValue().sum);
            out.writeDouble(entry.getValue().squaredSum);
            out.writeDouble(entry.getValue().tripleSum);
            out.writeDouble(entry.getValue().quarticSum);

            out.writeLong(entry.getValue().missingCount);
            out.writeLong(entry.getValue().totalCount);

            out.writeInt(entry.getValue().binCountPos.length);
            for(int i = 0; i < entry.getValue().binCountPos.length; i++) {
                out.writeLong(entry.getValue().binCountPos[i]);
            }

            out.writeInt(entry.getValue().binCountNeg.length);
            for(int i = 0; i < entry.getValue().binCountNeg.length; i++) {
                out.writeLong(entry.getValue().binCountNeg[i]);
            }

            out.writeInt(entry.getValue().binWeightPos.length);
            for(int i = 0; i < entry.getValue().binWeightPos.length; i++) {
                out.writeDouble(entry.getValue().binWeightPos[i]);
            }

            out.writeInt(entry.getValue().binWeightNeg.length);
            for(int i = 0; i < entry.getValue().binWeightNeg.length; i++) {
                out.writeDouble(entry.getValue().binWeightNeg[i]);
            }
        }
        out.writeBoolean(this.isEmpty);
    }

    @Override
    public void readFields(DataInput in) throws IOException {

        int mapSize = in.readInt();
        variableDailyStatInfo = new HashMap<>();
        for (int i = 0; i < mapSize; i++){
            int dateEntrySize = in.readInt();
            byte[] entryBytes = new byte[dateEntrySize];
            for(int k = 0; k < dateEntrySize; k++) {
                entryBytes[k] = in.readByte();
            }
            String dateMapKey = new String(entryBytes, Charset.forName("UTF-8"));
            VariableStatInfo info = new VariableStatInfo();
            variableDailyStatInfo.put(dateMapKey, info);

            info.setColumnConfigIndex(in.readInt());
            info.max = in.readDouble();
            info.min = in.readDouble();
            info.sum = in.readDouble();
            info.squaredSum = in.readDouble();
            info.tripleSum = in.readDouble();
            info.quarticSum = in.readDouble();

            info.missingCount = in.readLong();
            info.totalCount = in.readLong();

            int size = in.readInt();
            info.binCountPos = new long[size];
            for(int m = 0; m < size; m++) {
                info.binCountPos[m] = in.readLong();
            }

            size = in.readInt();
            info.binCountNeg = new long[size];
            for(int m = 0; m < size; m++) {
                info.binCountNeg[m] = in.readLong();
            }

            size = in.readInt();
            info.binWeightPos = new double[size];
            for(int m = 0; m < size; m++) {
                info.binWeightPos[m] = in.readDouble();
            }

            size = in.readInt();
            info.binWeightNeg = new double[size];
            for(int m = 0; m < size; m++) {
                info.binWeightNeg[m] = in.readDouble();
            }
        }

        this.isEmpty = in.readBoolean();
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


    public static class VariableStatInfo {

        private int columnConfigIndex;

        private double min = Double.MAX_VALUE;

        private double max = Double.MIN_VALUE;

        private double sum = 0.0d;

        private double mean = 0.0d;

        private double median = 0.0d;

        private double stdDev = 0.0d;

        private ColumnStatsCalculator.ColumnMetrics columnCountMetrics = null;

        private ColumnStatsCalculator.ColumnMetrics columnWeightMetrics = null;

        private double skewness = 0.0d;

        private double kurtosis = 0.0d;

        private long missingCount = 0L;

        private long totalCount = 0L;

        private double squaredSum = 0.0d;

        private double tripleSum = 0.0d;

        private double quarticSum = 0.0d;

        private long[] binCountPos;

        private long[] binCountNeg;

        private double[] binWeightPos;

        private double[] binWeightNeg;

        private long[] binCountTotal;

        private double p25th = 0d;

        private double p75th = 0d;

        public double getMean() {
            return mean;
        }

        public void setMean(double mean) {
            this.mean = mean;
        }

        public double getMedian() {
            return median;
        }

        public void setMedian(double median) {
            this.median = median;
        }

        public double getStdDev() {
            return stdDev;
        }

        public void setStdDev(double stdDev) {
            this.stdDev = stdDev;
        }

        public ColumnStatsCalculator.ColumnMetrics getColumnCountMetrics() {
            return columnCountMetrics;
        }

        public void setColumnCountMetrics(ColumnStatsCalculator.ColumnMetrics columnCountMetrics) {
            this.columnCountMetrics = columnCountMetrics;
        }

        public ColumnStatsCalculator.ColumnMetrics getColumnWeightMetrics() {
            return columnWeightMetrics;
        }

        public void setColumnWeightMetrics(ColumnStatsCalculator.ColumnMetrics columnWeightMetrics) {
            this.columnWeightMetrics = columnWeightMetrics;
        }

        public int getColumnConfigIndex() {
            return columnConfigIndex;
        }

        public void setColumnConfigIndex(int columnConfigIndex) {
            this.columnConfigIndex = columnConfigIndex;
        }

        public double getSkewness() {
            return skewness;
        }

        public void setSkewness(double skewness) {
            this.skewness = skewness;
        }

        public double getKurtosis() {
            return kurtosis;
        }

        public void setKurtosis(double kurtosis) {
            this.kurtosis = kurtosis;
        }

        public double getMin() {
            return min;
        }

        public void setMin(double min) {
            this.min = min;
        }

        public double getMax() {
            return max;
        }

        public void setMax(double max) {
            this.max = max;
        }

        public double getSum() {
            return sum;
        }

        public void setSum(double sum) {
            this.sum = sum;
        }

        public long getMissingCount() {
            return missingCount;
        }

        public void setMissingCount(long missingCount) {
            this.missingCount = missingCount;
        }

        public long getTotalCount() {
            return totalCount;
        }

        public void setTotalCount(long totalCount) {
            this.totalCount = totalCount;
        }

        public double getSquaredSum() {
            return squaredSum;
        }

        public void setSquaredSum(double squaredSum) {
            this.squaredSum = squaredSum;
        }

        public double getTripleSum() {
            return tripleSum;
        }

        public void setTripleSum(double tripleSum) {
            this.tripleSum = tripleSum;
        }

        public double getQuarticSum() {
            return quarticSum;
        }

        public void setQuarticSum(double quarticSum) {
            this.quarticSum = quarticSum;
        }

        public long[] getBinCountPos() {
            return binCountPos;
        }

        public void setBinCountPos(long[] binCountPos) {
            this.binCountPos = binCountPos;
        }

        public long[] getBinCountNeg() {
            return binCountNeg;
        }

        public void setBinCountNeg(long[] binCountNeg) {
            this.binCountNeg = binCountNeg;
        }

        public double[] getBinWeightPos() {
            return binWeightPos;
        }

        public void setBinWeightPos(double[] binWeightPos) {
            this.binWeightPos = binWeightPos;
        }

        public double[] getBinWeightNeg() {
            return binWeightNeg;
        }

        public long[] getBinCountTotal() {
            return binCountTotal;
        }

        public void setBinCountTotal(long[] binCountTotal) {
            this.binCountTotal = binCountTotal;
        }

        public void setBinWeightNeg(double[] binWeightNeg) {
            this.binWeightNeg = binWeightNeg;
        }

        public double getP25th() {
            return p25th;
        }

        public void setP25th(double p25th) {
            this.p25th = p25th;
        }

        public double getP75th() {
            return p75th;
        }

        public void setP75th(double p75th) {
            this.p75th = p75th;
        }

        public void init(int binSize){
            if(binCountPos == null){
                binCountPos = new long[binSize + 1];
            }
            if(binCountNeg == null){
                binCountNeg = new long[binSize + 1];
            }
            if(binWeightPos == null){
                binWeightPos = new double[binSize + 1];
            }
            if(binWeightNeg == null){
                binWeightNeg = new double[binSize + 1];
            }
        }

        @Override
        public String toString() {
            return "VariableStatInfo{" +
                    "columnConfigIndex=" + columnConfigIndex +
                    ", min=" + min +
                    ", max=" + max +
                    ", sum=" + sum +
                    ", mean=" + mean +
                    ", median=" + median +
                    ", stdDev=" + stdDev +
                    ", columnCountMetrics=" + columnCountMetrics +
                    ", columnWeightMetrics=" + columnWeightMetrics +
                    ", skewness=" + skewness +
                    ", kurtosis=" + kurtosis +
                    ", missingCount=" + missingCount +
                    ", totalCount=" + totalCount +
                    ", squaredSum=" + squaredSum +
                    ", tripleSum=" + tripleSum +
                    ", quarticSum=" + quarticSum +
                    ", binCountPos=" + Arrays.toString(binCountPos) +
                    ", binCountNeg=" + Arrays.toString(binCountNeg) +
                    ", binWeightPos=" + Arrays.toString(binWeightPos) +
                    ", binWeightNeg=" + Arrays.toString(binWeightNeg) +
                    ", binCountTotal=" + Arrays.toString(binCountTotal) +
                    ", p25th=" + p25th +
                    ", p75th=" + p75th +
                    '}';
        }
    }
}
