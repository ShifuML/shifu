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

import ml.shifu.guagua.GuaguaRuntimeException;

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
     * Do initialization like ModelConfig and ColumnConfig loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        this.outputKey = new IntWritable();
        this.outputValue = new Text();
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
                finalCw = cw;
            } else {
                finalCw.combine(cw);
            }
        }
        this.outputKey.set(key.get());
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

}