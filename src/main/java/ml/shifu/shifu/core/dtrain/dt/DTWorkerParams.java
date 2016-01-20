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

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DTWorkerParams implements Bytable {

    private Map<Integer, NodeStats> nodeStatsMap;

    public DTWorkerParams() {
    }

    public DTWorkerParams(Map<Integer, NodeStats> nodeStatsMap) {
        this.nodeStatsMap = nodeStatsMap;
    }

    @Override
    public void write(DataOutput out) throws IOException {
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
    public void readFields(DataInput in) throws IOException {
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

    public static class NodeStats implements Bytable {

        private int nodeId;

        private int treeId;

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
            out.writeInt(getNodeId());
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
            this.setNodeId(in.readInt());
            this.treeId = in.readInt();
            int len = in.readInt();
            this.featureStatistics = new HashMap<Integer, double[]>();
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
    }

}
