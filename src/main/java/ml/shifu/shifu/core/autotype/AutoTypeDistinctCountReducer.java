/*
 * Copyright [2012-2015] eBay Software Foundation
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
package ml.shifu.shifu.core.autotype;

import java.io.IOException;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;

/**
 * TODO
 */
public class AutoTypeDistinctCountReducer extends Reducer<IntWritable, BytesWritable, IntWritable, LongWritable> {

    private LongWritable outputValue = new LongWritable();

    @Override
    protected void reduce(IntWritable key, Iterable<BytesWritable> values, Context context) throws IOException,
            InterruptedException {
        HyperLogLogPlus hyperLogLogPlus = null;
        for(BytesWritable bytesWritable: values) {
            if(hyperLogLogPlus == null) {
                hyperLogLogPlus = HyperLogLogPlus.Builder.build(bytesWritable.getBytes());
            } else {
                try {
                    hyperLogLogPlus = (HyperLogLogPlus) hyperLogLogPlus.merge(HyperLogLogPlus.Builder
                            .build(bytesWritable.getBytes()));
                } catch (CardinalityMergeException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        outputValue.set(hyperLogLogPlus.cardinality());
        context.write(key, outputValue);
    }

}
