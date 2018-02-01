/*
 * Copyright [2013-2017] PayPal Software Foundation
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

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.GZIPOutputStream;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.collections.CollectionUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Binary neural network serializer.
 */
public class BinaryDTSerializer {

    private static final Logger LOG = LoggerFactory.getLogger(BinaryDTSerializer.class);

    /**
     * MARKER if we need to readUTF or readByte
     */
    public static final int UTF_BYTES_MARKER = -1;

    public static void save(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            List<List<TreeNode>> baggingTrees, String loss, int inputCount, FileSystem fs, Path output)
            throws IOException {
        LOG.info("Writing trees to {}.", output);
        save(modelConfig, columnConfigList, baggingTrees, loss, inputCount, fs.create(output));
    }

    public static void save(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            List<List<TreeNode>> baggingTrees, String loss, int inputCount, OutputStream output) throws IOException {
        DataOutputStream fos = null;

        try {
            fos = new DataOutputStream(new GZIPOutputStream(output));
            // version
            fos.writeInt(CommonConstants.TREE_FORMAT_VERSION);
            fos.writeUTF(modelConfig.getAlgorithm());
            fos.writeUTF(loss);
            fos.writeBoolean(modelConfig.isClassification());
            fos.writeBoolean(modelConfig.getTrain().isOneVsAll());
            fos.writeInt(inputCount);

            Map<Integer, String> columnIndexNameMapping = new HashMap<Integer, String>();
            Map<Integer, List<String>> columnIndexCategoricalListMapping = new HashMap<Integer, List<String>>();
            Map<Integer, Double> numericalMeanMapping = new HashMap<Integer, Double>();
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.isFinalSelect()) {
                    columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
                }
                if(columnConfig.isCategorical() && CollectionUtils.isNotEmpty(columnConfig.getBinCategory())) {
                    columnIndexCategoricalListMapping.put(columnConfig.getColumnNum(), columnConfig.getBinCategory());
                }

                if(columnConfig.isNumerical() && columnConfig.getMean() != null) {
                    numericalMeanMapping.put(columnConfig.getColumnNum(), columnConfig.getMean());
                }
            }

            if(columnIndexNameMapping.size() == 0) {
                boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
                for(ColumnConfig columnConfig: columnConfigList) {
                    if(CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                        columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
                    }
                }
            }

            // serialize numericalMeanMapping
            fos.writeInt(numericalMeanMapping.size());
            for(Entry<Integer, Double> entry: numericalMeanMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                // for some feature, it is null mean value, it is not selected, just set to 0d to avoid NPE
                fos.writeDouble(entry.getValue() == null ? 0d : entry.getValue());
            }
            // serialize columnIndexNameMapping
            fos.writeInt(columnIndexNameMapping.size());
            for(Entry<Integer, String> entry: columnIndexNameMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeUTF(entry.getValue());
            }
            // serialize columnIndexCategoricalListMapping
            fos.writeInt(columnIndexCategoricalListMapping.size());
            for(Entry<Integer, List<String>> entry: columnIndexCategoricalListMapping.entrySet()) {
                List<String> categories = entry.getValue();
                if(categories != null) {
                    fos.writeInt(entry.getKey());
                    fos.writeInt(categories.size());
                    for(String category: categories) {
                        // There is 16k limitation when using writeUTF() function.
                        // if the category value is larger than 10k, write a marker -1 and write bytes instead of
                        // writeUTF;
                        // in read part logic should be changed also to readByte not readUTF according to the marker
                        if(category.length() < Constants.MAX_CATEGORICAL_VAL_LEN) {
                            fos.writeUTF(category);
                        } else {
                            fos.writeShort(UTF_BYTES_MARKER); // marker here
                            byte[] bytes = category.getBytes("UTF-8");
                            fos.writeInt(bytes.length);
                            for(int i = 0; i < bytes.length; i++) {
                                fos.writeByte(bytes[i]);
                            }
                        }
                    }
                }
            }

            Map<Integer, Integer> columnMapping = getColumnMapping(columnConfigList);
            fos.writeInt(columnMapping.size());
            for(Entry<Integer, Integer> entry: columnMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeInt(entry.getValue());
            }

            // after model version 4 (>=4), IndependentTreeModel support bagging, here write a default RF/GBT size 1
            fos.writeInt(baggingTrees.size());
            for(int i = 0; i < baggingTrees.size(); i++) {
                List<TreeNode> trees = baggingTrees.get(i);
                int treeLength = trees.size();
                fos.writeInt(treeLength);
                for(TreeNode treeNode: trees) {
                    treeNode.write(fos);
                }
            }
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private static Map<Integer, Integer> getColumnMapping(List<ColumnConfig> columnConfigList) {
        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnConfigList.size(), 1f);
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
        boolean isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        int index = 0;
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(!isAfterVarSelect) {
                if(!columnConfig.isMeta() && !columnConfig.isTarget()
                        && CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            } else {
                if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                        && columnConfig.isFinalSelect()) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            }
        }
        return columnMapping;
    }

}
