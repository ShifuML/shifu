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
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;

/**
 * Worker result return to master.
 * 
 * <p>
 * The first part is for error collections: {@link #count} and {@link #squareError}.
 * 
 * <p>
 * {@link #nodeStatsMap} includes node statistics for each node, key is node group index id from master.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 * 
 * @see NodeStats
 */
public class DTWorkerParams extends HaltBytable implements Combinable<DTWorkerParams> {

    /**
     * # of records per such worker.
     */
    private long count;

    /**
     * Error for such worker and such iteration.
     */
    private double squareError;

    /**
     * Node statistic map including node group index and node stats object.
     */
    private Map<Integer, NodeStats> nodeStatsMap;

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "DTWorkerParams [count=" + count + ", squareError=" + squareError + ", nodeStatsMap=" + nodeStatsMap
                + "]";
    }

    public DTWorkerParams() {
    }

    public DTWorkerParams(long count, double squareError, Map<Integer, NodeStats> nodeStatsMap) {
        this.count = count;
        this.squareError = squareError;
        this.nodeStatsMap = nodeStatsMap;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        out.writeLong(count);
        out.writeDouble(squareError);
        if(nodeStatsMap == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeInt(nodeStatsMap.size());
            for(Entry<Integer, NodeStats> entry: nodeStatsMap.entrySet()) {
                out.writeInt(entry.getKey());
                entry.getValue().write(out);
            }
        }
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        this.count = in.readLong();
        this.squareError = in.readDouble();
        if(in.readBoolean()) {
            this.nodeStatsMap = new HashMap<Integer, NodeStats>();
            int len = in.readInt();
            for(int i = 0; i < len; i++) {
                int key = in.readInt();
                NodeStats stats = new NodeStats();
                stats.readFields(in);
                this.nodeStatsMap.put(key, stats);
            }
        }
    }

    /**
     * @return the nodeStatsMap
     */
    public Map<Integer, NodeStats> getNodeStatsMap() {
        return nodeStatsMap;
    }

    /**
     * @param nodeStatsMap
     *            the nodeStatsMap to set
     */
    public void setNodeStatsMap(Map<Integer, NodeStats> nodeStatsMap) {
        this.nodeStatsMap = nodeStatsMap;
    }

    /**
     * @return the count
     */
    public long getCount() {
        return count;
    }

    /**
     * @return the squareError
     */
    public double getSquareError() {
        return squareError;
    }

    /**
     * @param count
     *            the count to set
     */
    public void setCount(long count) {
        this.count = count;
    }

    /**
     * @param squareError
     *            the squareError to set
     */
    public void setSquareError(double squareError) {
        this.squareError = squareError;
    }

    /**
     * Node statistics with {@link #featureStatistics} including all statistics for all sub-sampling features.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    public static class NodeStats implements Bytable {

        /**
         * Node id in one node.
         */
        private int nodeId;

        /**
         * Tree id for such node.
         */
        private int treeId;

        /**
         * Feature statistics for sub-sampling features.
         */
        private Map<Integer, double[]> featureStatistics;

        public NodeStats() {
        }

        public NodeStats(int treeId, int nodeId, Map<Integer, double[]> featureStatistics) {
            this.treeId = treeId;
            this.nodeId = nodeId;
            this.featureStatistics = featureStatistics;
        }

        /**
         * @return the treeId
         */
        public int getTreeId() {
            return treeId;
        }

        /**
         * @return the featureStatistics
         */
        public Map<Integer, double[]> getFeatureStatistics() {
            return featureStatistics;
        }

        /**
         * @param treeIndex
         *            the treeIndex to set
         */
        public void setTreeId(int treeId) {
            this.treeId = treeId;
        }

        /**
         * @param featureStatistics
         *            the featureStatistics to set
         */
        public void setFeatureStatistics(Map<Integer, double[]> featureStatistics) {
            this.featureStatistics = featureStatistics;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(nodeId);
            out.writeInt(treeId);
            out.writeInt(this.featureStatistics.size());
            for(Entry<Integer, double[]> entry: this.featureStatistics.entrySet()) {
                out.writeInt(entry.getKey());
                out.writeInt(entry.getValue().length);
                for(int i = 0; i < entry.getValue().length; i++) {
                    out.writeDouble(entry.getValue()[i]);
                }
            }
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            this.nodeId = in.readInt();
            this.treeId = in.readInt();
            int len = in.readInt();
            this.featureStatistics = new HashMap<Integer, double[]>(len, 1f);
            for(int i = 0; i < len; i++) {
                int key = in.readInt();
                int vLen = in.readInt();
                double[] values = new double[vLen];
                for(int j = 0; j < vLen; j++) {
                    values[j] = in.readDouble();
                }
                this.featureStatistics.put(key, values);
            }
        }

        /**
         * @return the nodeId
         */
        public int getNodeId() {
            return nodeId;
        }

        /**
         * @param nodeId
         *            the nodeId to set
         */
        public void setNodeId(int nodeId) {
            this.nodeId = nodeId;
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#toString()
         */
        @Override
        public String toString() {
            return "NodeStats [nodeId=" + nodeId + ", treeId=" + treeId + ", featureStatistics=" + featureStatistics
                    + "]";
        }
    }

    @Override
    public DTWorkerParams combine(DTWorkerParams that) {
        assert that != null;

        this.count += that.count;
        this.squareError += that.squareError;

        if(this.nodeStatsMap != null && that.nodeStatsMap != null) {
            for(Entry<Integer, NodeStats> entry: this.nodeStatsMap.entrySet()) {
                NodeStats nodeStats = entry.getValue();
                NodeStats thatNodeStats = that.nodeStatsMap.get(entry.getKey());
                assert nodeStats.nodeId == thatNodeStats.nodeId;
                assert nodeStats.treeId == thatNodeStats.treeId;

                for(Entry<Integer, double[]> featureStatsEntry: nodeStats.getFeatureStatistics().entrySet()) {
                    double[] thisFeatureStats = featureStatsEntry.getValue();
                    double[] thatFeatureStats = thatNodeStats.featureStatistics.get(featureStatsEntry.getKey());
                    for(int i = 0; i < thisFeatureStats.length; i++) {
                        thisFeatureStats[i] += thatFeatureStats[i];
                    }
                }
            }
        }

        return this;
    }

}
