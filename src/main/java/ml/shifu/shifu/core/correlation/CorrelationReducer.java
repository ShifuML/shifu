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

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.codec.binary.Base64;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * {@link CorrelationReducer} is used to merge all {@link CorrelationWritable}s together to compute pearson correlation
 * between two variables.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class CorrelationReducer extends Reducer<IntWritable, CorrelationWritable, IntWritable, Text> {

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * Prevent too many new objects for output key.
     */
    private Text outputValue;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    @SuppressWarnings("unused")
    private boolean hasCandidates = false;

    /**
     * Do initialization like ModelConfig and ColumnConfig loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        this.outputKey = new IntWritable();
        this.outputValue = new Text();
        loadConfigFiles(context);
    }

    private void loadConfigFiles(final Context context) {
        try {
            SourceType sourceType = SourceType.valueOf(context.getConfiguration().get(
                    Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);
            this.hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected void reduce(IntWritable key, Iterable<CorrelationWritable> values, Context context) throws IOException,
            InterruptedException {
        // build final correlation column info
        CorrelationWritable finalCw = null;

        Iterator<CorrelationWritable> cwIt = values.iterator();
        while(cwIt.hasNext()) {
            CorrelationWritable cw = cwIt.next();
            if(finalCw == null) {
                finalCw = initCw(cw.getAdjustCount().length);
            }
            finalCw.setColumnIndex(cw.getColumnIndex());
            finalCw.combine(cw);
        }

        this.outputKey.set(key.get());
        this.outputValue.set(new String(Base64.encodeBase64(objectToBytes(finalCw)), "utf-8"));
        context.write(outputKey, outputValue);
    }

    private CorrelationWritable initCw(int statsCnt) {
        CorrelationWritable finalCw = new CorrelationWritable();
        double[] xySum = new double[statsCnt];
        finalCw.setXySum(xySum);
        double[] xxSum = new double[statsCnt];
        finalCw.setXxSum(xxSum);
        double[] yySum = new double[statsCnt];
        finalCw.setYySum(yySum);
        double[] adjustCount = new double[statsCnt];
        finalCw.setAdjustCount(adjustCount);
        double[] adjustSumX = new double[statsCnt];
        finalCw.setAdjustSumX(adjustSumX);
        double[] adjustSumY = new double[statsCnt];
        finalCw.setAdjustSumY(adjustSumY);
        return finalCw;
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

}