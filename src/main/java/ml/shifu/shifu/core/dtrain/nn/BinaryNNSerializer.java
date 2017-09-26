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
package ml.shifu.shifu.core.dtrain.nn;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.GZIPOutputStream;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.encog.ml.BasicML;

/**
 * Binary neural network serializer.
 */
public class BinaryNNSerializer {

    public static void save(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, List<BasicML> basicNetworks,
            FileSystem fs, Path output) throws IOException {
        DataOutputStream fos = null;
        try {
            fos = new DataOutputStream(new GZIPOutputStream(fs.create(output)));

            // version
            fos.writeInt(CommonConstants.NN_FORMAT_VERSION);

            // write normStr
            String normStr = modelConfig.getNormalize().getNormType().toString();
            ml.shifu.shifu.core.dtrain.StringUtils.writeString(fos, normStr);

            // compute columns needed
            Map<Integer, String> columnIndexNameMapping = getIndexNameMapping(columnConfigList);

            // write column stats to output
            List<NNColumnStats> csList = new ArrayList<NNColumnStats>();
            for(ColumnConfig cc: columnConfigList) {
                if(columnIndexNameMapping.containsKey(cc.getColumnNum())) {
                    NNColumnStats cs = new NNColumnStats();
                    cs.setCutoff(modelConfig.getNormalizeStdDevCutOff());
                    cs.setColumnType(cc.getColumnType());
                    cs.setMean(cc.getMean());
                    cs.setStddev(cc.getStdDev());
                    cs.setColumnNum(cc.getColumnNum());
                    cs.setColumnName(cc.getColumnName());
                    cs.setBinCategories(cc.getBinCategory());
                    cs.setBinBoundaries(cc.getBinBoundary());
                    cs.setBinPosRates(cc.getBinPosRate());
                    cs.setBinCountWoes(cc.getBinCountWoe());
                    cs.setBinWeightWoes(cc.getBinWeightedWoe());

                    // TODO cache such computation
                    double[] meanAndStdDev = Normalizer.calculateWoeMeanAndStdDev(cc, false);
                    cs.setWoeMean(meanAndStdDev[0]);
                    cs.setWoeStddev(meanAndStdDev[1]);
                    double[] WgtMeanAndStdDev = Normalizer.calculateWoeMeanAndStdDev(cc, true);
                    cs.setWoeWgtMean(WgtMeanAndStdDev[0]);
                    cs.setWoeWgtStddev(WgtMeanAndStdDev[1]);

                    csList.add(cs);
                }
            }

            fos.writeInt(csList.size());
            for(NNColumnStats cs: csList) {
                cs.write(fos);
            }

            // write column index mapping
            Map<Integer, Integer> columnMapping = getColumnMapping(columnConfigList);
            fos.writeInt(columnMapping.size());
            for(Entry<Integer, Integer> entry: columnMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeInt(entry.getValue());
            }

            // persist network, set it as list
            fos.writeInt(basicNetworks.size());
            for(BasicML network: basicNetworks) {
                new PersistBasicFloatNetwork().saveNetwork(fos, (BasicFloatNetwork) network);
            }
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private static Map<Integer, String> getIndexNameMapping(List<ColumnConfig> columnConfigList) {
        Map<Integer, String> columnIndexNameMapping = new HashMap<Integer, String>();
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
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
        return columnIndexNameMapping;
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
