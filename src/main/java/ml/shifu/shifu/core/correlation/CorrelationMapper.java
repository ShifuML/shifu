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

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link CorrelationMapper} is used to compute {@link CorrelationWritable} per column per mapper.
 * 
 * <p>
 * Such {@link CorrelationWritable} is sent to reducer (only one) to merge and compute real pearson value.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class CorrelationMapper extends Mapper<LongWritable, Text, IntWritable, CorrelationWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(CorrelationMapper.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    private String dataSetDelimiter;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * To filter records by customized expressions
     */
    private DataPurifier dataPurifier;

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * Correlation map with <column_idm columnInfo>
     */
    private Map<Integer, CorrelationWritable> correlationMap;

    /**
     * Count in current mapper
     */
    private long count;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * For categorical feature, a map is used to save query time in execution
     */
    private Map<Integer, Map<String, Integer>> categoricalIndexMap = new HashMap<Integer, Map<String, Integer>>();

    /**
     * If compute all pairs (i, j), if false, only computes pairs (i, j) when i >= j
     */
    private boolean isComputeAll = false;

    // cache tags in set for search
    protected Set<String> posTagSet;
    protected Set<String> negTagSet;
    protected Set<String> tagSet;
    private List<Set<String>> tags;

    private void loadConfigFiles(final Context context) {
        try {
            SourceType sourceType = SourceType.valueOf(context.getConfiguration().get(
                    Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(
                    context.getConfiguration().get(Constants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.dataSetDelimiter = this.modelConfig.getDataSetDelimiter();

        this.dataPurifier = new DataPurifier(this.modelConfig);

        this.isComputeAll = Boolean.valueOf(context.getConfiguration().get(Constants.SHIFU_CORRELATION_COMPUTE_ALL,
                "false"));

        this.outputKey = new IntWritable();
        this.correlationMap = new HashMap<Integer, CorrelationWritable>();

        for(ColumnConfig config: columnConfigList) {
            if(config.isCategorical()) {
                Map<String, Integer> map = new HashMap<String, Integer>();
                if(config.getBinCategory() != null) {
                    for(int i = 0; i < config.getBinCategory().size(); i++) {
                        map.put(config.getBinCategory().get(i), i);
                    }
                }
                this.categoricalIndexMap.put(config.getColumnNum(), map);
            }
        }

        if(modelConfig != null && modelConfig.getPosTags() != null) {
            this.posTagSet = new HashSet<String>(modelConfig.getPosTags());
        }
        if(modelConfig != null && modelConfig.getNegTags() != null) {
            this.negTagSet = new HashSet<String>(modelConfig.getNegTags());
        }
        if(modelConfig != null && modelConfig.getFlattenTags() != null) {
            this.tagSet = new HashSet<String>(modelConfig.getFlattenTags());
        }

        this.tags = this.modelConfig.getSetTags();
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }
        double[] dValues = null;
        if(!this.dataPurifier.isFilterOut(valueStr)) {
            return;
        }

        context.getCounter(Constants.SHIFU_GROUP_COUNTER, "CNT_AFTER_FILTER").increment(1L);

        if(Math.random() >= this.modelConfig.getStats().getSampleRate()) {
            return;
        }

        context.getCounter(Constants.SHIFU_GROUP_COUNTER, "CORRELATION_CNT").increment(1L);

        dValues = getDoubleArrayByRawArray(CommonUtils.split(valueStr, this.dataSetDelimiter));

        count += 1L;
        if(count % 2000L == 0) {
            LOG.info("Current records: {} in thread {}.", count, Thread.currentThread().getName());
        }

        for(int i = 0; i < this.columnConfigList.size(); i++) {
            ColumnConfig columnConfig = this.columnConfigList.get(i);
            if(columnConfig.getColumnFlag() == ColumnFlag.Meta) {
                continue;
            }
            CorrelationWritable cw = this.correlationMap.get(i);
            if(cw == null) {
                cw = new CorrelationWritable();
                this.correlationMap.put(i, cw);
            }
            cw.setColumnIndex(i);
            cw.setCount(cw.getCount() + 1d);
            cw.setSum(cw.getSum() + dValues[i]);
            double squaredSum = dValues[i] * dValues[i];
            cw.setSumSquare(cw.getSumSquare() + squaredSum);
            double[] xySum = cw.getXySum();
            if(xySum == null) {
                xySum = new double[this.columnConfigList.size()];
                cw.setXySum(xySum);
            }
            double[] xxSum = cw.getXxSum();
            if(xxSum == null) {
                xxSum = new double[this.columnConfigList.size()];
                cw.setXxSum(xxSum);
            }
            double[] yySum = cw.getYySum();
            if(yySum == null) {
                yySum = new double[this.columnConfigList.size()];
                cw.setYySum(yySum);
            }

            double[] adjustCount = cw.getAdjustCount();
            if(adjustCount == null) {
                adjustCount = new double[this.columnConfigList.size()];
                cw.setAdjustCount(adjustCount);
            }
            double[] adjustSumX = cw.getAdjustSumX();
            if(adjustSumX == null) {
                adjustSumX = new double[this.columnConfigList.size()];
                cw.setAdjustSumX(adjustSumX);
            }
            double[] adjustSumY = cw.getAdjustSumY();
            if(adjustSumY == null) {
                adjustSumY = new double[this.columnConfigList.size()];
                cw.setAdjustSumY(adjustSumY);
            }

            for(int j = 0; j < this.columnConfigList.size(); j++) {
                ColumnConfig otherColumnConfig = this.columnConfigList.get(j);
                if(otherColumnConfig.getColumnFlag() == ColumnFlag.Meta) {
                    continue;
                }
                if(i > j && !this.isComputeAll) {
                    continue;
                }
                // only do stats on both valid value
                if(dValues[i] != Double.MIN_VALUE && dValues[j] != Double.MIN_VALUE) {
                    xySum[j] += dValues[i] * dValues[j];
                    xxSum[j] += squaredSum;
                    yySum[j] += dValues[j] * dValues[j];
                    adjustCount[j] += 1d;
                    adjustSumX[j] += dValues[i];
                    adjustSumY[j] += dValues[j];
                }
            }
        }
    }

    private double[] getDoubleArrayByRawArray(String[] units) {
        double[] dValues = new double[this.columnConfigList.size()];
        for(int i = 0; i < this.columnConfigList.size(); i++) {
            ColumnConfig columnConfig = this.columnConfigList.get(i);
            if(columnConfig.getColumnFlag() == ColumnFlag.Meta) {
                // only meta columns not in correlation
                dValues[i] = 0d;
            } else if(columnConfig.getColumnFlag() == ColumnFlag.Target) {
                if(this.tagSet.contains(units[i])) {
                    if(modelConfig.isRegression()) {
                        if(this.posTagSet.contains(units[i])) {
                            dValues[i] = 1d;
                        }
                        if(this.negTagSet.contains(units[i])) {
                            dValues[i] = 0d;
                        }
                    } else {
                        int index = -1;
                        for(int j = 0; j < tags.size(); j++) {
                            Set<String> tagSet = tags.get(j);
                            if(tagSet.contains(units[0])) {
                                index = j;
                                break;
                            }
                        }
                        dValues[i] = index;
                    }
                } else {
                    // Invalid target
                    dValues[i] = Double.MIN_VALUE;
                }
            } else {
                if(columnConfig.isNumerical()) {
                    // if missing it is set to MIN_VALUE, then try to skip rows with invalid value
                    if(units[i] == null || units[i].length() == 0) {
                        // some null values, set it to min value to avoid parsing String to improve performance
                        dValues[i] = Double.MIN_VALUE;
                    } else {
                        dValues[i] = NumberFormatUtils.getDouble(units[i], Double.MIN_VALUE);
                    }
                }
                if(columnConfig.isCategorical()) {
                    if(columnConfig.getBinCategory() == null) {
                        if(System.currentTimeMillis() % 100L == 0) {
                            LOG.warn(
                                    "Column {} with null binCategory but is not meta or target column, set to 0d for correlation.",
                                    columnConfig.getColumnName());
                        }
                        dValues[i] = 0d;
                        continue;
                    }
                    Integer index = null;
                    if(units[i] != null) {
                        index = this.categoricalIndexMap.get(columnConfig.getColumnNum()).get(units[i]);
                    }
                    if(index == null || index == -1) {
                        dValues[i] = columnConfig.getBinPosRate().get(columnConfig.getBinPosRate().size() - 1);
                    } else {
                        Double binPosRate = columnConfig.getBinPosRate().get(index);
                        if(binPosRate == null) {
                            dValues[i] = columnConfig.getBinPosRate().get(columnConfig.getBinPosRate().size() - 1);
                        } else {
                            dValues[i] = binPosRate;
                        }
                    }
                }
            }
        }
        return dValues;
    }

    /**
     * Write column info to reducer for merging.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        LOG.info("Final records in such thread of mapper: {}.", count);
        for(Entry<Integer, CorrelationWritable> entry: this.correlationMap.entrySet()) {
            outputKey.set(entry.getKey());
            context.write(outputKey, entry.getValue());
        }
    }
}
