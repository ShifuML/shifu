/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.posttrain;

import java.io.IOException;
import java.util.List;

import ml.shifu.shifu.core.posttrain.FeatureStatsWritable.BinStats;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link PostTrainReducer} is to aggregate sum of score per each bin of each variable together to compute average score
 * value.
 * 
 * <p>
 * Only 1 reducer is OK, since all mappers are feature-wised and 1 reducer is enough to process all variables. 
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class PostTrainReducer extends Reducer<IntWritable, FeatureStatsWritable, IntWritable, Text> {

    private Text outputValue = null;

    private final static Logger LOG = LoggerFactory.getLogger(PostTrainReducer.class);

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        this.outputValue = new Text();
    }

    @Override
    protected void reduce(IntWritable key, Iterable<FeatureStatsWritable> values, Context context) throws IOException,
            InterruptedException {
        List<BinStats> binStats = null;
        for(FeatureStatsWritable fsw: values) {
            if(binStats == null) {
                binStats = fsw.getBinStats();
            } else {
                for(int i = 0; i < binStats.size(); i++) {
                    BinStats rbs = binStats.get(i);
                    BinStats bs = fsw.getBinStats().get(i);
                    rbs.setBinSum(rbs.getBinSum() + bs.getBinSum());
                    rbs.setBinCnt(rbs.getBinCnt() + bs.getBinCnt());
                }
            }
        }

        StringBuilder sb = new StringBuilder(150);
        for(int i = 0; i < binStats.size(); i++) {
            BinStats bs = binStats.get(i);
            int avgScore = 0;
            if(bs.getBinCnt() != 0L) {
                avgScore = (int) (bs.getBinSum() / bs.getBinCnt());
            }
            if(i == binStats.size() - 1) {
                sb.append(avgScore);
            } else {
                sb.append(avgScore).append(',');
            }
        }
        LOG.info(key.toString() + " " + sb.toString());
        this.outputValue.set(sb.toString());
        context.write(key, this.outputValue);
    }

}
