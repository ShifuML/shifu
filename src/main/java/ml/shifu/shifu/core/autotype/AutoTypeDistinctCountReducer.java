/*
 * Copyright [2012-2015] PayPal Software Foundation
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
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;

/**
 * To merge all mapper {@link HyperLogLogPlus} statistics together according to variable id.
 */
public class AutoTypeDistinctCountReducer extends
        Reducer<IntWritable, CountAndFrequentItemsWritable, IntWritable, Text> {

    private Text outputValue = new Text();

    @Override
    protected void reduce(IntWritable key, Iterable<CountAndFrequentItemsWritable> values, Context context)
            throws IOException, InterruptedException {
        HyperLogLogPlus hyperLogLogPlus = null;
        Set<String> fis = new HashSet<String>();
        long count = 0, invalidCount = 0, validNumCount = 0;
        for(CountAndFrequentItemsWritable cfiw: values) {
            count += cfiw.getCount();
            invalidCount += cfiw.getInvalidCount();
            validNumCount += cfiw.getValidNumCount();
            fis.addAll(cfiw.getFrequetItems());
            if(hyperLogLogPlus == null) {
                hyperLogLogPlus = HyperLogLogPlus.Builder.build(cfiw.getHyperBytes());
            } else {
                try {
                    hyperLogLogPlus = (HyperLogLogPlus) hyperLogLogPlus.merge(HyperLogLogPlus.Builder.build(cfiw
                            .getHyperBytes()));
                } catch (CardinalityMergeException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        outputValue.set(count + ":" + invalidCount + ":" + validNumCount + ":" + hyperLogLogPlus.cardinality() + ":"
                + limitedFrequentItems(fis));
        context.write(key, outputValue);
    }

    private static String limitedFrequentItems(Set<String> fis) {
        StringBuilder sb = new StringBuilder(200);
        int size = Math.min(fis.size(), CountAndFrequentItemsWritable.FREQUET_ITEM_MAX_SIZE * 10);
        Iterator<String> iterator = fis.iterator();
        int i = 0;
        while(i < size) {
            String next = iterator.next().replaceAll(":", " ").replace(",", " ");
            sb.append(next);
            if(i != size - 1) {
                sb.append(",");
            }
            i += 1;
        }
        return sb.toString();
    }
}
