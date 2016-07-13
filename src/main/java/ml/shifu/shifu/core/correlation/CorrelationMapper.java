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
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
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
     * Model Config read from HDFS
     */
    @SuppressWarnings("unused")
    private ModelConfig modelConfig;

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * Correlation map with <column_idm columnInfo>
     */
    private Map<Integer, CorrelationWritable> correlationMap;

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

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.outputKey = new IntWritable();
        this.correlationMap = new HashMap<Integer, CorrelationWritable>();
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        // StringUtils.isBlank is not used here to avoid import new jar
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }

        double[] dValues = getDoubleArray(CommonUtils.split(valueStr, Constants.DEFAULT_DELIMITER));
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
            for(int j = 0; j < this.columnConfigList.size(); j++) {
                ColumnConfig otherColumnConfig = this.columnConfigList.get(j);
                if(otherColumnConfig.isMeta() || otherColumnConfig.isTarget()) {
                    continue;
                }
                xySum[j] += dValues[i] * dValues[j];
            }
        }
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
