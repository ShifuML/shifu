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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * {@link FeatureImportanceReducer} is to aggregate feature importance statistics and compute the top important
 * variables.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class FeatureImportanceReducer extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {

    private Map<Integer, Double> variableStatsMap = new HashMap<Integer, Double>();

    public static class FeatureScore {

        public FeatureScore(int columnNum, double binAvgScore) {
            this.columnNum = columnNum;
            this.binAvgScore = binAvgScore;
        }

        private int columnNum;
        private double binAvgScore;
    }

    @Override
    protected void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context) throws IOException,
            InterruptedException {
        double sum = 0d;
        for(DoubleWritable dw: values) {
            sum += dw.get();
        }
        this.variableStatsMap.put(key.get(), sum);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        List<FeatureScore> featureScores = new ArrayList<FeatureImportanceReducer.FeatureScore>();
        for(Entry<Integer, Double> entry: variableStatsMap.entrySet()) {
            featureScores.add(new FeatureScore(entry.getKey(), entry.getValue()));
        }

        Collections.sort(featureScores, new Comparator<FeatureScore>() {
            @Override
            public int compare(FeatureScore fs1, FeatureScore fs2) {
                if(fs1.binAvgScore < fs2.binAvgScore) {
                    return 1;
                }
                if(fs1.binAvgScore > fs2.binAvgScore) {
                    return -1;
                }

                return 0;
            }
        });

        for(FeatureScore featureScore: featureScores) {
            context.write(new IntWritable(featureScore.columnNum), new DoubleWritable(featureScore.binAvgScore));
        }
    }

}
