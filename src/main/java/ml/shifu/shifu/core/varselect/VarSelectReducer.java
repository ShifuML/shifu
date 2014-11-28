/*
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.varselect;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * {@link VarSelectReducer} is used to accumulate all mapper column-MSE values together.
 * 
 * <p>
 * This is a global accumulation, reducer number in current MapReduce job should be set to 1.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class VarSelectReducer extends Reducer<LongWritable, DoubleWritable, LongWritable, NullWritable> {

    // private final static Logger LOG = LoggerFactory.getLogger(VarSelectReducer.class);

    /**
     * Final results list, this list is loaded in memory for sum, and will be written by context in cleanup.
     */
    private List<Pair> results = new ArrayList<Pair>();

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Basic input node count for NN model, all the variables selected in current model training.
     */
    private int inputNodeCount;

    /**
     * To set as a ratio instead an absolute number, each time it is
     * a ratio. For example, 100 variables, using ratio 0.05, first time select 95 variables, next as candidates are
     * decreasing, next time it is still 0.05, but only 4 variables are removed.
     */
    private float wrapperRatio;

    /**
     * Prevent too many new objects for output key.
     */
    private LongWritable outputKey;

    /**
     * Wrapper by adding(A), removing(R) or sensitivity(SE).
     */
    private String wrapperBy;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
    private void loadConfigFiles(final Context context) {
        try {
            SourceType sourceType = SourceType.valueOf(context.getConfiguration().get(
                    Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
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
        int[] inputOutputIndex = getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.wrapperRatio = context.getConfiguration().getFloat(Constants.SHIFU_VARSELECT_WRAPPER_RATIO, 0.05f);
        this.outputKey = new LongWritable();

        this.wrapperBy = context.getConfiguration().get(Constants.SHIFU_VARSELECT_WRAPPER_TYPE, "SE");
    }

    @Override
    protected void reduce(LongWritable key, Iterable<DoubleWritable> values, Context context) throws IOException,
            InterruptedException {
        double MSE = 0d;
        for(DoubleWritable value: values) {
            MSE += value.get();
        }
        results.add(new Pair(key.get(), MSE));
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        Collections.sort(this.results, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.valueOf(o2.value).compareTo(Double.valueOf(o1.value));
            }
        });

        int candidates = 0;
        if("R".equalsIgnoreCase(this.wrapperBy) || "SE".equalsIgnoreCase(this.wrapperBy)) {
            candidates = (int) (this.inputNodeCount * (1.0f - this.wrapperRatio));
        } else {
            // wrapper by A
            candidates = (int) (this.inputNodeCount * (this.wrapperRatio));
        }

        for(int i = 0; i < candidates; i++) {
            this.outputKey.set(this.results.get(i).key);
            context.write(this.outputKey, NullWritable.get());
        }

    }

    /**
     * Get input nodes number (final select) and output nodes number from column config, and candidate input node
     * number.
     * 
     * <p>
     * If number of column in final-select is 0, which means to select all non meta and non target columns. So the input
     * number is set to all candidates.
     * 
     * @throws NullPointerException
     *             if columnConfigList or ColumnConfig object in columnConfigList is null.
     */
    private static int[] getInputOutputCandidateCounts(List<ColumnConfig> columnConfigList) {
        int input = 0, output = 0, candidate = 0;
        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta()) {
                candidate++;
            }
            if(config.isFinalSelect()) {
                input++;
            }
            if(config.isTarget()) {
                output++;
            }
        }
        return new int[] { input, output, candidate };
    }

    private static class Pair {

        public Pair(long key, double value) {
            this.key = key;
            this.value = value;
        }

        public long key;
        public double value;

        @Override
        public String toString() {
            return key + ":" + value;
        }
    }

}
