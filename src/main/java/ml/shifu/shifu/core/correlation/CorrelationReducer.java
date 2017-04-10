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

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.Correlation;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.codec.binary.Base64;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
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
    @SuppressWarnings("unused")
    private SortedMap<Integer, CorrelationWritable> correlationMap;

    /**
     * Correlation type
     */
    @SuppressWarnings("unused")
    private Correlation correlation;

    /**
     * If compute all pairs (i, j), if false, only computes pairs (i, j) when i >= j
     */
    @SuppressWarnings("unused")
    private boolean isComputeAll = false;

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
        for(ColumnConfig config: columnConfigList) {
            // set to null to avoid big memory consumption, correlation values are not used, GC will free memory.
            config.setCorrArray(null);
        }
        this.outputKey = new IntWritable();
        this.outputValue = new Text();
        this.correlationMap = new TreeMap<Integer, CorrelationWritable>();
        this.correlation = this.modelConfig.getNormalize().getCorrelation();
        this.isComputeAll = Boolean.valueOf(context.getConfiguration().get(Constants.SHIFU_CORRELATION_COMPUTE_ALL,
                "false"));
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
        this.outputKey.set(key.get());
        Base64.encodeBase64(objectToBytes(finalCw));
        this.outputValue.set(new String(Base64.encodeBase64(objectToBytes(finalCw)), "utf-8"));
        context.write(outputKey, outputValue);
    }

    public byte[] objectToBytes(Writable result) {
        ByteArrayOutputStream out = null;
        DataOutputStream dataOut = null;
        try {
            out = new ByteArrayOutputStream();
            dataOut = new DataOutputStream(out);
            result.write(dataOut);
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        } finally {
            if(dataOut != null) {
                try {
                    dataOut.close();
                } catch (IOException e) {
                    throw new GuaguaRuntimeException(e);
                }
            }
        }
        return out.toByteArray();
    }

    // /**
    // * Write column info to output.
    // */
    // @Override
    // protected void cleanup(Context context) throws IOException, InterruptedException {
    // SortedMap<Integer, double[]> corrDoubleArrayMap = new TreeMap<Integer, double[]>();
    // for(Entry<Integer, CorrelationWritable> entry: this.correlationMap.entrySet()) {
    // outputKey.set(entry.getKey());
    // ColumnConfig xColumnConfig = this.columnConfigList.get(entry.getKey());
    // if(xColumnConfig.getColumnFlag() == ColumnFlag.Meta) {
    // continue;
    // }
    // CorrelationWritable xCw = this.correlationMap.get(entry.getKey());
    // double[] corrArray = new double[this.columnConfigList.size()];
    // for(int i = 0; i < corrArray.length; i++) {
    // ColumnConfig yColumnConfig = this.columnConfigList.get(i);
    // if(yColumnConfig.getColumnFlag() == ColumnFlag.Meta) {
    // continue;
    // }
    // CorrelationWritable yCw = this.correlationMap.get(i);
    // if(entry.getKey() > i && !this.isComputeAll) {
    // double[] reverseDoubleArray = corrDoubleArrayMap.get(i);
    // if(reverseDoubleArray != null) {
    // corrArray[i] = reverseDoubleArray[entry.getKey()];
    // } else {
    // corrArray[i] = 0d;
    // }
    // // not compute all, only up-right matrix are computed, such case, just get [i, j] from [j, i]
    // continue;
    // }
    //
    // if(correlation == Correlation.Pearson) {
    // // Count*Sum(X*Y) - SUM(X)*SUM(Y)
    // if(xCw == null) {
    // LOG.info("xCw is null with index: {}", entry.getKey());
    // continue;
    // }
    //
    // if(yCw == null) {
    // LOG.info("yCw is null with index: {}", i);
    // continue;
    // }
    //
    // double numerator = xCw.getAdjustCount()[i] * xCw.getXySum()[i] - xCw.getAdjustSum()[i]
    // * yCw.getAdjustSum()[i];
    // // Math.sqrt ( COUNT * SUM(X2) - SUM(X) * SUM(X) ) * Math.sqrt ( COUNT * SUM(Y2) - SUM(Y) * SUM(Y) )
    // double denominator1 = Math.sqrt(xCw.getAdjustCount()[i] * xCw.getAdjustSumSquare()[i]
    // - xCw.getAdjustSum()[i] * xCw.getAdjustSum()[i]);
    // double denominator2 = Math.sqrt(yCw.getAdjustCount()[i] * yCw.getAdjustSumSquare()[i]
    // - yCw.getAdjustSum()[i] * yCw.getAdjustSum()[i]);
    // if(Double.compare(denominator1, Double.valueOf(0d)) == 0
    // || Double.compare(denominator2, Double.valueOf(0d)) == 0) {
    // corrArray[i] = 0d;
    // } else {
    // corrArray[i] = numerator / (denominator1 * denominator2);
    // }
    // } else if(correlation == Correlation.NormPearson) {
    // // Count*Sum(X*Y) - SUM(X)*SUM(Y)
    // double numerator = xCw.getCount() * xCw.getXySum()[i] - xCw.getSum() * yCw.getSum();
    // // Math.sqrt ( COUNT * SUM(X2) - SUM(X) * SUM(X) ) * Math.sqrt ( COUNT * SUM(Y2) - SUM(Y) * SUM(Y) )
    // double denominator1 = Math.sqrt(xCw.getCount() * xCw.getSumSquare() - xCw.getSum() * xCw.getSum());
    // double denominator2 = Math.sqrt(yCw.getCount() * yCw.getSumSquare() - yCw.getSum() * yCw.getSum());
    // if(Double.compare(denominator1, Double.valueOf(0d)) == 0
    // || Double.compare(denominator2, Double.valueOf(0d)) == 0) {
    // corrArray[i] = 0d;
    // } else {
    // corrArray[i] = numerator / (denominator1 * denominator2);
    // }
    // }
    // }
    // outputValue.set(Arrays.toString(corrArray));
    // corrDoubleArrayMap.put(entry.getKey(), corrArray);
    // context.write(outputKey, outputValue);
    // }
    // }

}