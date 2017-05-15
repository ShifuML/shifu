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
package ml.shifu.shifu.core.posttrain;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Writable;

/**
 * {@link Writable} implementation for bin average score. Sum and count values are included in bin list of one feature..
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class FeatureStatsWritable implements Writable {

    public static class BinStats {
        private double binSum;
        private long binCnt;

        public BinStats(double binSum, long binCnt) {
            this.setBinSum(binSum);
            this.setBinCnt(binCnt);
        }

        /**
         * @return the binSum
         */
        public double getBinSum() {
            return binSum;
        }

        /**
         * @param binSum
         *            the binSum to set
         */
        public void setBinSum(double binSum) {
            this.binSum = binSum;
        }

        /**
         * @return the binCnt
         */
        public long getBinCnt() {
            return binCnt;
        }

        /**
         * @param binCnt
         *            the binCnt to set
         */
        public void setBinCnt(long binCnt) {
            this.binCnt = binCnt;
        }
    }

    public FeatureStatsWritable() {
    }

    public FeatureStatsWritable(List<BinStats> binStats) {
        this.binStats = binStats;
    }

    private List<BinStats> binStats;

    /**
     * @return the binStats
     */
    public List<BinStats> getBinStats() {
        return binStats;
    }

    /**
     * @param binStats
     *            the binStats to set
     */
    public void setBinStats(List<BinStats> binStats) {
        this.binStats = binStats;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        if(binStats == null) {
            out.writeInt(0);
        } else {
            out.writeInt(binStats.size());
            for(BinStats bin: binStats) {
                out.writeDouble(bin.getBinSum());
                out.writeLong(bin.getBinCnt());
            }
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        int len = in.readInt();
        this.binStats = new ArrayList<FeatureStatsWritable.BinStats>(len);
        for(int i = 0; i < len; i++) {
            this.binStats.add(new BinStats(in.readDouble(), in.readLong()));
        }
    }

}
