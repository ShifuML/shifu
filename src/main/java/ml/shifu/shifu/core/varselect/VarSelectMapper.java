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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.guagua.util.MemoryUtils;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.encog.ml.MLRegression;
import org.encog.ml.data.basic.BasicMLData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * Mapper implementation to accumulate MSE value when remove one column.
 * 
 * <p>
 * All the MSE values are accumulated in in-memory HashMap {@link #results}, which will also be write out in
 * {@link #cleanup(org.apache.hadoop.mapreduce.Mapper.Context)}.
 * 
 * <p>
 * Output of all the mappers will be read and accumulated in VarSelectReducer to get all global MSE values. In Reducer,
 * all MSE values sorted and select valid variables.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class VarSelectMapper extends Mapper<LongWritable, Text, LongWritable, ColumnInfo> {

    private final static Logger LOG = LoggerFactory.getLogger(VarSelectMapper.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    private static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR);

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Basic neural network model instance to compute basic score with all selected columns and wrapper selected
     * columns.
     */
    private MLRegression model;

    /**
     * Basic input node count for NN model, all the variables selected in current model training.
     */
    private int inputNodeCount;

    /**
     * Final results map, this map is loaded in memory for sum, and will be written by context in cleanup.
     */
    private Map<Long, ColumnInfo> results = new HashMap<Long, ColumnInfo>();

    /**
     * Inputs columns for each record. To save new objects in
     * {@link #map(LongWritable, Text, org.apache.hadoop.mapreduce.Mapper.Context)}.
     */
    private double[] inputs;

    /**
     * Outputs columns for each record. To save new objects in
     * {@link #map(LongWritable, Text, org.apache.hadoop.mapreduce.Mapper.Context)}.
     */
    private double[] outputs;

    /**
     * Column indexes for each record. To save new objects in
     * {@link #map(LongWritable, Text, org.apache.hadoop.mapreduce.Mapper.Context)}.
     */
    private long[] columnIndexes;

    /**
     * Input MLData instance to save new.
     */
    private BasicMLData inputsMLData;

    /**
     * Prevent too many new objects for output key.
     */
    private LongWritable outputKey;

    /**
     * Filter by sensitivity by target(ST) or sensitivity(SE).
     */
    private String filterBy;

    /**
     * A counter to count # of records in current mapper.
     */
    private long recordCount;

    private Set<Integer> featureSet;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
    private void loadConfigFiles(final Context context) {
        LOG.info("Before loading config with memory {} in thread {}.", MemoryUtils.getRuntimeMemoryStats(), Thread
                .currentThread().getName());
        long start = System.currentTimeMillis();
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
        LOG.info("After loading config with time {}ms and memory {} in thread {}.",
                (System.currentTimeMillis() - start), MemoryUtils.getRuntimeMemoryStats(), Thread.currentThread()
                        .getName());
    }

    /**
     * Load first model in model path as a {@link MLRegression} instance.
     */
    private void loadModel() throws IOException {
        LOG.info("Before loading model with memory {} in thread {}.", MemoryUtils.getRuntimeMemoryStats(), Thread
                .currentThread().getName());
        long start = System.currentTimeMillis();
        this.model = (MLRegression) (CommonUtils.loadBasicModels(this.modelConfig, this.columnConfigList, null).get(0));
        LOG.info("After load model with time {}ms and memory {} in thread {}.", (System.currentTimeMillis() - start),
                MemoryUtils.getRuntimeMemoryStats(), Thread.currentThread().getName());
    }

    /**
     * Do initialization like ModelConfig and ColumnConfig loading, model loading and others like input or output number
     * loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        loadModel();
        this.filterBy = context.getConfiguration()
                .get(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE, Constants.FILTER_BY_SE);
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        if(this.model instanceof BasicFloatNetwork) {
            this.inputs = new double[((BasicFloatNetwork) this.model).getFeatureSet().size()];
            this.featureSet = ((BasicFloatNetwork) this.model).getFeatureSet();
        } else {
            this.inputs = new double[this.inputNodeCount];
        }

        boolean isAfterVarSelect = (inputOutputIndex[0] != 0);
        // cache all feature list for sampling features
        if(this.featureSet == null || this.featureSet.size() == 0) {
            this.featureSet = new HashSet<Integer>(CommonUtils.getAllFeatureList(columnConfigList, isAfterVarSelect));
            this.inputs = new double[this.featureSet.size()];
        }
        this.outputs = new double[inputOutputIndex[1]];
        this.columnIndexes = new long[this.inputs.length];
        this.inputsMLData = new BasicMLData(this.inputs.length);
        this.outputKey = new LongWritable();
        LOG.info("Filter by is {}", filterBy);
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        recordCount += 1L;
        int index = 0, inputsIndex = 0, outputsIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(value.toString())) {
            double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
            if(index == this.columnConfigList.size()) {
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    this.outputs[outputsIndex++] = doubleValue;
                } else {
                    if(this.featureSet.contains(columnConfig.getColumnNum())) {
                        inputs[inputsIndex] = doubleValue;
                        columnIndexes[inputsIndex++] = columnConfig.getColumnNum();
                    }
                }
            }
            index++;
        }

        double oldValue = 0.0d;

        this.inputsMLData.setData(this.inputs);

        double candidateModelScore = 0d;
        if(Constants.FILTER_BY_SE.equalsIgnoreCase(this.filterBy)) {
            candidateModelScore = this.model.compute(new BasicMLData(inputs)).getData()[0];
        }
        for(int i = 0; i < this.inputs.length; i++) {
            oldValue = this.inputs[i];
            this.inputs[i] = 0d;
            this.inputsMLData.setData(this.inputs);
            double currentModelScore = this.model.compute(new BasicMLData(inputs)).getData()[0];

            double diff = 0d;
            if(Constants.FILTER_BY_ST.equalsIgnoreCase(this.filterBy)) {
                // ST
                diff = this.outputs[0] - currentModelScore;
            } else {
                // SE
                diff = candidateModelScore - currentModelScore;
            }
            ColumnInfo columnInfo = this.results.get(this.columnIndexes[i]);

            if(columnInfo == null) {
                columnInfo = new ColumnInfo();
                columnInfo.setSumScoreDiff(Math.abs(diff));
                columnInfo.setSumSquareScoreDiff(power2(diff));
            } else {
                columnInfo.setSumScoreDiff(columnInfo.getSumScoreDiff() + Math.abs(diff));
                columnInfo.setSumSquareScoreDiff(columnInfo.getSumSquareScoreDiff() + power2(diff));
            }
            this.results.put(this.columnIndexes[i], columnInfo);
            this.inputs[i] = oldValue;
        }

        if(this.recordCount % 1000 == 0) {
            LOG.info("Finish to process {} records.", this.recordCount);
        }
    }

    /**
     * Write all column-&gt;MSE pairs to output.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Entry<Long, ColumnInfo> entry: results.entrySet()) {
            this.outputKey.set(entry.getKey());
            // value is sumValue, not sumValue/(number of records)
            ColumnInfo columnInfo = entry.getValue();
            columnInfo.setCount(this.recordCount);
            context.write(this.outputKey, columnInfo);
        }
        LOG.debug("Final results: {}", results);
    }

    private double power2(double data) {
        return data * data;
    }

}
