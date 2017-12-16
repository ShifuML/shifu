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
import java.util.Set;

import ml.shifu.guagua.util.MemoryUtils;
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
     * Model Config read from HDFS, be static to shared in multiple mappers
     */
    private static ModelConfig modelConfig;

    /**
     * To filter records by customized expressions
     */
    private DataPurifier dataPurifier;

    /**
     * Count in current mapper
     */
    private long count;

    /**
     * Column Config list read from HDFS, be static to shared in multiple mappers
     */
    private static List<ColumnConfig> columnConfigList;

    /**
     * For categorical feature, a map is used to save query time in execution
     */
    private Map<Integer, Map<String, Integer>> categoricalIndexMap = new HashMap<Integer, Map<String, Integer>>();

    /**
     * If compute all pairs (i, j), if false, only computes pairs (i, j) when i >= j
     */
    private boolean isComputeAll = false;

    private static boolean hasCandidates = false;

    // cache tags in set for search
    protected Set<String> posTagSet;
    protected Set<String> negTagSet;
    protected Set<String> tagSet;
    private List<Set<String>> tags;

    private synchronized static void loadConfigFiles(final Context context) {
        if(modelConfig == null) {
            LOG.info("Before loading config with memory {} in thread {}.", MemoryUtils.getRuntimeMemoryStats(), Thread
                    .currentThread().getName());
            long start = System.currentTimeMillis();
            try {
                modelConfig = CommonUtils.loadModelConfig(Constants.MODEL_CONFIG_JSON_FILE_NAME, SourceType.LOCAL);
                columnConfigList = CommonUtils.loadColumnConfigList(Constants.COLUMN_CONFIG_JSON_FILE_NAME,
                        SourceType.LOCAL);
                hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            LOG.info("After loading config with time {}ms and memory {} in thread {}.",
                    (System.currentTimeMillis() - start), MemoryUtils.getRuntimeMemoryStats(), Thread.currentThread()
                            .getName());
        }
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.dataSetDelimiter = modelConfig.getDataSetDelimiter();

        this.dataPurifier = new DataPurifier(modelConfig);

        this.isComputeAll = Boolean.valueOf(context.getConfiguration().get(Constants.SHIFU_CORRELATION_COMPUTE_ALL,
                "false"));

        for(ColumnConfig config: columnConfigList) {
            if(config.isCategorical()) {
                Map<String, Integer> map = new HashMap<String, Integer>();
                if(config.getBinCategory() != null) {
                    for(int i = 0; i < config.getBinCategory().size(); i++) {
                        List<String> cvals = CommonUtils.flattenCatValGrp(config.getBinCategory().get(i));
                        for(String cval: cvals) {
                            map.put(cval, i);
                        }
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

        this.tags = modelConfig.getSetTags();
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }
        double[] dValues = null;
        if(!this.dataPurifier.isFilter(valueStr)) {
            return;
        }

        long startO = System.currentTimeMillis();

        context.getCounter(Constants.SHIFU_GROUP_COUNTER, "CNT_AFTER_FILTER").increment(1L);

        // make sampling work in correlation
        if(Math.random() >= modelConfig.getStats().getSampleRate()) {
            return;
        }

        context.getCounter(Constants.SHIFU_GROUP_COUNTER, "CORRELATION_CNT").increment(1L);

        dValues = getDoubleArrayByRawArray(CommonUtils.split(valueStr, this.dataSetDelimiter));

        count += 1L;
        if(count % 2000L == 0) {
            LOG.info("Current records: {} in thread {}.", count, Thread.currentThread().getName());
        }

        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(columnConfig.getColumnFlag() == ColumnFlag.Meta
                    || (hasCandidates && !ColumnFlag.Candidate.equals(columnConfig.getColumnFlag()))) {
                continue;
            }

            CorrelationWritable cw = CorrelationMultithreadedMapper.finalCorrelationMap.get(i);
            synchronized(cw) {
                cw.setColumnIndex(i);
                cw.setCount(cw.getCount() + 1d);
                cw.setSum(cw.getSum() + dValues[i]);
                double squaredSum = dValues[i] * dValues[i];
                cw.setSumSquare(cw.getSumSquare() + squaredSum);
                double[] xySum = cw.getXySum();
                if(xySum == null) {
                    xySum = new double[columnConfigList.size()];
                    cw.setXySum(xySum);
                }
                double[] xxSum = cw.getXxSum();
                if(xxSum == null) {
                    xxSum = new double[columnConfigList.size()];
                    cw.setXxSum(xxSum);
                }
                double[] yySum = cw.getYySum();
                if(yySum == null) {
                    yySum = new double[columnConfigList.size()];
                    cw.setYySum(yySum);
                }

                double[] adjustCount = cw.getAdjustCount();
                if(adjustCount == null) {
                    adjustCount = new double[columnConfigList.size()];
                    cw.setAdjustCount(adjustCount);
                }
                double[] adjustSumX = cw.getAdjustSumX();
                if(adjustSumX == null) {
                    adjustSumX = new double[columnConfigList.size()];
                    cw.setAdjustSumX(adjustSumX);
                }
                double[] adjustSumY = cw.getAdjustSumY();
                if(adjustSumY == null) {
                    adjustSumY = new double[columnConfigList.size()];
                    cw.setAdjustSumY(adjustSumY);
                }

                for(int j = 0; j < columnConfigList.size(); j++) {
                    ColumnConfig otherColumnConfig = columnConfigList.get(j);
                    if(otherColumnConfig.getColumnFlag() == ColumnFlag.Meta) {
                        continue;
                    }
                    if(i > j && !this.isComputeAll) {
                        continue;
                    }

                    // only do stats on both valid values
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
            LOG.debug("running time is {}ms in thread {}", (System.currentTimeMillis() - startO), Thread
                    .currentThread().getName());
        }
    }

    private double[] getDoubleArrayByRawArray(String[] units) {
        double[] dValues = new double[columnConfigList.size()];
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
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
    }

}
