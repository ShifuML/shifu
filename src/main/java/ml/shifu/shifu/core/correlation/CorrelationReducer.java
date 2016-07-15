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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.Correlation;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link CorrelationReducer} is used to merge all {@link CorrelationWritable}s together to compute pearson correlation
 * between two variables.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class CorrelationReducer extends Reducer<IntWritable, CorrelationWritable, IntWritable, Text> {

    @SuppressWarnings("unused")
    private final static Logger LOG = LoggerFactory.getLogger(CorrelationReducer.class);

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * Prevent too many new objects for output key.
     */
    private Text outputValue;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Correlation map with <column_idm columnInfo>
     */
    private Map<Integer, CorrelationWritable> correlationMap;

    private Correlation correlation;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
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
     * Do initialization like ModelConfig and ColumnConfig loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);
        this.outputKey = new IntWritable();
        this.outputValue = new Text();
        this.correlationMap = new HashMap<Integer, CorrelationWritable>();
        this.correlation = this.modelConfig.getNormalize().getCorrelation();
    }

    @Override
    protected void reduce(IntWritable key, Iterable<CorrelationWritable> values, Context context) throws IOException,
            InterruptedException {
        // build final correlation column info
        CorrelationWritable finalCw = new CorrelationWritable();
        finalCw.setColumnIndex(key.get());
        finalCw.setXySum(new double[this.columnConfigList.size()]);
        finalCw.setXxSum(new double[this.columnConfigList.size()]);
        finalCw.setYySum(new double[this.columnConfigList.size()]);
        finalCw.setAdjustCount(new double[this.columnConfigList.size()]);
        finalCw.setAdjustSum(new double[this.columnConfigList.size()]);
        finalCw.setAdjustSumSquare(new double[this.columnConfigList.size()]);

        Iterator<CorrelationWritable> cwIt = values.iterator();
        while(cwIt.hasNext()) {
            CorrelationWritable cw = cwIt.next();
            finalCw.setCount(finalCw.getCount() + cw.getCount());
            finalCw.setSum(finalCw.getSum() + cw.getSum());
            finalCw.setSumSquare(finalCw.getSumSquare() + cw.getSumSquare());
            double[] finalXySum = finalCw.getXySum();
            double[] xySum = cw.getXySum();
            for(int i = 0; i < finalXySum.length; i++) {
                finalXySum[i] += xySum[i];
            }
            double[] finalXxSum = finalCw.getXxSum();
            double[] xxSum = cw.getXxSum();
            for(int i = 0; i < finalXxSum.length; i++) {
                finalXxSum[i] += xxSum[i];
            }
            double[] finalYySum = finalCw.getYySum();
            double[] yySum = cw.getYySum();
            for(int i = 0; i < finalYySum.length; i++) {
                finalYySum[i] += yySum[i];
            }

            double[] finalAdjustCount = finalCw.getAdjustCount();
            double[] adjustCount = cw.getAdjustCount();
            for(int i = 0; i < finalAdjustCount.length; i++) {
                finalAdjustCount[i] += adjustCount[i];
            }
            double[] finalAdjustSum = finalCw.getAdjustSum();
            double[] adjustSum = cw.getAdjustSum();
            for(int i = 0; i < finalAdjustSum.length; i++) {
                finalAdjustSum[i] += adjustSum[i];
            }
            double[] finalAdjustSumSquare = finalCw.getAdjustSumSquare();
            double[] adjustSumSquare = cw.getAdjustSumSquare();
            for(int i = 0; i < finalAdjustSumSquare.length; i++) {
                finalAdjustSumSquare[i] += adjustSumSquare[i];
            }
        }
        this.correlationMap.put(key.get(), finalCw);
    }

    /**
     * Write column info to output.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Entry<Integer, CorrelationWritable> entry: this.correlationMap.entrySet()) {
            outputKey.set(entry.getKey());
            ColumnConfig xColumnConfig = this.columnConfigList.get(entry.getKey());
            if(xColumnConfig.isMeta() || xColumnConfig.isTarget()) {
                continue;
            }
            CorrelationWritable xCw = this.correlationMap.get(entry.getKey());
            double[] corrArray = new double[this.columnConfigList.size()];
            for(int i = 0; i < corrArray.length; i++) {
                ColumnConfig yColumnConfig = this.columnConfigList.get(i);
                if(yColumnConfig.isMeta() || yColumnConfig.isTarget()) {
                    continue;
                }
                CorrelationWritable yCw = this.correlationMap.get(i);
                if(correlation == Correlation.Pearson) {
                    // Count*Sum(X*Y) - SUM(X)*SUM(Y)
                    double numerator = xCw.getAdjustCount()[i] * xCw.getXySum()[i] - xCw.getAdjustSum()[i]
                            * yCw.getAdjustSum()[i];
                    // Math.sqrt ( COUNT * SUM(X2) - SUM(X) * SUM(X) ) * Math.sqrt ( COUNT * SUM(Y2) - SUM(Y) * SUM(Y) )
                    double denominator1 = Math.sqrt(xCw.getAdjustCount()[i] * xCw.getAdjustSumSquare()[i]
                            - xCw.getAdjustSum()[i] * xCw.getAdjustSum()[i]);
                    double denominator2 = Math.sqrt(yCw.getAdjustCount()[i] * yCw.getAdjustSumSquare()[i]
                            - yCw.getAdjustSum()[i] * yCw.getAdjustSum()[i]);
                    if(Double.compare(denominator1, Double.valueOf(0d)) == 0
                            || Double.compare(denominator2, Double.valueOf(0d)) == 0) {
                        corrArray[i] = 0d;
                    } else {
                        corrArray[i] = numerator / (denominator1 * denominator2);
                    }
                } else if(correlation == Correlation.NormPearson) {
                    // Count*Sum(X*Y) - SUM(X)*SUM(Y)
                    double numerator = xCw.getCount() * xCw.getXySum()[i] - xCw.getSum() * yCw.getSum();
                    // Math.sqrt ( COUNT * SUM(X2) - SUM(X) * SUM(X) ) * Math.sqrt ( COUNT * SUM(Y2) - SUM(Y) * SUM(Y) )
                    double denominator1 = Math.sqrt(xCw.getCount() * xCw.getSumSquare() - xCw.getSum() * xCw.getSum());
                    double denominator2 = Math.sqrt(yCw.getCount() * yCw.getSumSquare() - yCw.getSum() * yCw.getSum());
                    if(Double.compare(denominator1, Double.valueOf(0d)) == 0
                            || Double.compare(denominator2, Double.valueOf(0d)) == 0) {
                        corrArray[i] = 0d;
                    } else {
                        corrArray[i] = numerator / (denominator1 * denominator2);
                    }
                }
            }
            outputValue.set(Arrays.toString(corrArray));
            context.write(outputKey, outputValue);
        }
    }

}