/*
 * Copyright [2012-2014] PayPal Software Foundation
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
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link VarSelectReducer} is used to accumulate all mapper column-MSE values together.
 * 
 * <p>
 * This is a global accumulation, reducer number in current MapReduce job should be set to 1.
 * 
 * <p>
 * Input type is (ColumnId, Iterable(MSE)) from all mapper tasks.
 * 
 * <p>
 * In {@link #cleanup(org.apache.hadoop.mapreduce.Reducer.Context)}, variables with MSE will be sorted according to
 * variable wrapper type. According to {@link #filterOutRatio} setting, only variables in that range will be written
 * into HDFS.
 * 
 * <p>
 * {@link #filterOutRatio} means each time we need remove how many percentage of variables. A ratio is better than a
 * fixed number. Since each time we reduce variables which number is also decreased. Say 100 variables, wrapperRatio is
 * 0.05. First time we remove 100*0.05 = 5 variables, second time 95 * 0.05 variables will be removed.
 * 
 * <p>
 * TODO Add mean value, not only MSE value; Write mean and MSE to files for later analysis.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class VarSelectReducer extends Reducer<LongWritable, ColumnInfo, Text, Text> {

    private final static Logger LOG = LoggerFactory.getLogger(VarSelectReducer.class);

    /**
     * Final results list, this list is loaded in memory for sum, and will be written by context in cleanup.
     */
    private List<Pair> results = new ArrayList<Pair>();

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

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
    private float filterOutRatio;

    /**
     * Explicit set number of variables to be selected,this overwrites filterOutRatio
     */
    private int filterNum;

    /**
     * Prevent too many new objects for output key.
     */
    private Text outputKey;

    /**
     * Prevent too many new objects for output key.
     */
    private Text outputValue;

    /**
     * Output value text.
     */
    private final static Text OUTPUT_VALUE = new Text("");

    /**
     * Wrapper by sensitivity by target(ST) or sensitivity(SE).
     */
    private String filterBy;

    /**
     * Multiple outputs to write se report in HDFS.
     */
    private MultipleOutputs<Text, Text> mos;

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

        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(), this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.filterOutRatio = context.getConfiguration().getFloat(Constants.SHIFU_VARSELECT_FILTEROUT_RATIO,
                Constants.SHIFU_DEFAULT_VARSELECT_FILTEROUT_RATIO);
        this.filterNum = context.getConfiguration().getInt(Constants.SHIFU_VARSELECT_FILTER_NUM,
                Constants.SHIFU_DEFAULT_VARSELECT_FILTER_NUM);
        this.outputKey = new Text();
        this.outputValue = new Text();
        this.filterBy = context.getConfiguration()
                .get(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE, Constants.FILTER_BY_SE);
        this.mos = new MultipleOutputs<Text, Text>(context);
        LOG.info("FilterBy is {}, filterOutRatio is {}, filterNum is {}", filterBy, filterOutRatio, filterNum);
    }

    @Override
    protected void reduce(LongWritable key, Iterable<ColumnInfo> values, Context context) throws IOException,
            InterruptedException {
        ColumnStatistics column = new ColumnStatistics();
        double sum = 0d;
        double sumSquare = 0d;
        long count = 0L;
        for(ColumnInfo info: values) {
            sum += info.getSumScoreDiff();
            sumSquare += info.getSumSquareScoreDiff();
            count += info.getCount();
        }
        column.setMean(sum / count);
        column.setRms(Math.sqrt(sumSquare / count));
        column.setVariance((sumSquare / count) - power2(sum / count));
        this.results.add(new Pair(key.get(), column));
    }

    private double power2(double data) {
        return data * data;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        Collections.sort(this.results, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.compare(o2.value.getRms(), o1.value.getRms());
            }
        });

        LOG.debug("Final Results:{}", this.results);

        int candidates = this.filterNum;
        if(candidates <= 0) {
            if(Constants.FILTER_BY_ST.equalsIgnoreCase(this.filterBy)
                    || Constants.FILTER_BY_SE.equalsIgnoreCase(this.filterBy)) {
                candidates = (int) (this.inputNodeCount * (1.0f - this.filterOutRatio));
            } else {
                // wrapper by A
                candidates = (int) (this.inputNodeCount * (this.filterOutRatio));
            }
        }

        LOG.info("Candidates count is {}", candidates);

        for(int i = 0; i < this.results.size(); i++) {
            Pair pair = this.results.get(i);
            this.outputKey.set(pair.key + "");
            if(i < candidates) {
                context.write(this.outputKey, OUTPUT_VALUE);
            }
            // for thousands of features, here using 'new' ok
            StringBuilder sb = new StringBuilder(100);
            sb.append(this.columnConfigList.get((int) pair.key).getColumnName()).append("\t")
                    .append(pair.value.getMean()).append("\t").append(pair.value.getRms()).append("\t")
                    .append(pair.value.getVariance());
            this.outputValue.set(sb.toString());
            this.mos.write(Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME, this.outputKey, this.outputValue);
        }

        this.mos.close();
    }

    private static class Pair {

        public Pair(long key, ColumnStatistics value) {
            this.key = key;
            this.value = value;
        }

        public long key;
        public ColumnStatistics value;

        @Override
        public String toString() {
            return key + ":" + value;
        }
    }

}
