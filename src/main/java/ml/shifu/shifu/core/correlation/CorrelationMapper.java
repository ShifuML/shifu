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
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.Correlation;
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
     * Weight column index.
     */
    private int weightedColumnNum = -1;

    /**
     * Tag column index
     */
    private int tagColumnNum = -1;

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * Correlation map with <column_idm columnInfo>
     */
    private Map<Integer, CorrelationWritable> correlationMap;

    private Correlation correlation;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

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

    /**
     * Load tag weight index field.
     */
    private void loadTagWeightNum() {
        for(ColumnConfig config: this.columnConfigList) {
            if(config.isTarget()) {
                this.tagColumnNum = config.getColumnNum();
                break;
            }
        }

        if(this.tagColumnNum == -1) {
            throw new RuntimeException("No valid target column.");
        }
    }

    /**
     * Load weight column index field.
     */
    private void loadWeightColumnNum() {
        String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();
        if(weightColumnName != null && weightColumnName.length() != 0) {
            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                if(config.getColumnName().equals(weightColumnName)) {
                    this.weightedColumnNum = i;
                    break;
                }
            }
        }
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.dataSetDelimiter = this.modelConfig.getDataSetDelimiter();

        this.dataPurifier = new DataPurifier(this.modelConfig);

        loadWeightColumnNum();

        loadTagWeightNum();

        this.outputKey = new IntWritable();
        this.correlationMap = new HashMap<Integer, CorrelationWritable>();
        this.correlation = modelConfig.getNormalize().getCorrelation();
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }
        double[] dValues = null;
        if(correlation == Correlation.Pearson) {
            if(!this.dataPurifier.isFilterOut(valueStr)) {
                return;
            }
            dValues = getDoubleArrayByRawArray(CommonUtils.split(valueStr, this.dataSetDelimiter));
        } else if(correlation == Correlation.NormPearson) {
            dValues = getDoubleArray(CommonUtils.split(valueStr, Constants.DEFAULT_DELIMITER));
        }
        for(int i = 0; i < this.columnConfigList.size(); i++) {
            ColumnConfig columnConfig = this.columnConfigList.get(i);
            if(columnConfig.isMeta() || columnConfig.isTarget()) {
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
            cw.setSumSquare(cw.getSumSquare() + dValues[i] * dValues[i]);
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
                cw.setYySum(xxSum);
            }

            double[] adjustCount = cw.getAdjustCount();
            if(adjustCount == null) {
                adjustCount = new double[this.columnConfigList.size()];
                cw.setAdjustCount(adjustCount);
            }
            double[] adjustSum = cw.getAdjustSum();
            if(adjustSum == null) {
                adjustSum = new double[this.columnConfigList.size()];
                cw.setAdjustSum(adjustSum);
            }
            double[] adjustSumSquare = cw.getAdjustSumSquare();
            if(adjustSumSquare == null) {
                adjustSumSquare = new double[this.columnConfigList.size()];
                cw.setAdjustSumSquare(adjustSumSquare);
            }

            for(int j = 0; j < this.columnConfigList.size(); j++) {
                ColumnConfig otherColumnConfig = this.columnConfigList.get(j);
                if(otherColumnConfig.isMeta() || otherColumnConfig.isTarget()) {
                    continue;
                }
                if(Double.compare(dValues[i], Double.MIN_VALUE) != 0
                        && Double.compare(dValues[j], Double.MIN_VALUE) != 0) {
                    xySum[j] += dValues[i] * dValues[j];
                    xxSum[j] += dValues[i] * dValues[i];
                    yySum[j] += dValues[j] * dValues[j];
                    adjustCount[j] += 1d;
                    adjustSum[j] += dValues[i];
                    adjustSumSquare[j] += dValues[i] * dValues[i];
                }
            }
        }
    }

    private double[] getDoubleArrayByRawArray(String[] units) {
        double[] dValues = new double[this.columnConfigList.size()];
        for(int i = 0; i < this.columnConfigList.size(); i++) {
            ColumnConfig columnConfig = this.columnConfigList.get(i);
            if(columnConfig.isMeta() || columnConfig.isTarget()
                    || columnConfig.getColumnNum() == this.weightedColumnNum) {
                dValues[i] = 0d;
            } else {
                if(columnConfig.isNumerical()) {
                    // if missing it is set to MIN_VALUE, then try to skip rows between
                    dValues[i] = NumberFormatUtils.getDouble(units[i], Double.MIN_VALUE);
                }
                if(columnConfig.isCategorical()) {
                    if(columnConfig.getBinCategory() == null) {
                        if(System.currentTimeMillis() % 50L == 0) {
                            LOG.warn("Column "
                                    + columnConfig.getColumnName()
                                    + " with null binCategory but is not meta or target column, set to 0d for correlation.");
                        }
                        dValues[i] = 0d;
                        continue;
                    }
                    int index = -1;
                    if(units[i] != null) {
                        // TODO use set to replace indexOf
                        index = columnConfig.getBinCategory().indexOf(units[i]);
                    }
                    if(index == -1) {
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

    private double[] getDoubleArray(String[] units) {
        double[] dValues = new double[this.columnConfigList.size()];
        for(int i = 0; i < this.columnConfigList.size(); i++) {
            ColumnConfig columnConfig = this.columnConfigList.get(i);
            if(columnConfig.isMeta() || columnConfig.isTarget()) {
                dValues[i] = 0d;
            } else {
                dValues[i] = NumberFormatUtils.getDouble(units[i], 0d);
            }
        }
        return dValues;
    }

    /**
     * Write column info to reducer for merging.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Entry<Integer, CorrelationWritable> entry: this.correlationMap.entrySet()) {
            outputKey.set(entry.getKey());
            context.write(outputKey, entry.getValue());
        }
    }

}
