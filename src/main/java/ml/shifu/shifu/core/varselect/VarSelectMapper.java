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
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.CommonUtils.FileSuffixPathFilter;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
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
public class VarSelectMapper extends Mapper<LongWritable, Text, LongWritable, DoubleWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(VarSelectMapper.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    private static final Splitter DEFAULT_SPLITTER = Splitter.on(NNConstants.NN_DEFAULT_COLUMN_SEPARATOR);

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
     * 
     * TODO so far only NN is supported, think about how to extend to SVM and LR.
     */
    private BasicNetwork model;

    /**
     * Basic input node count for NN model, all the variables selected in current model training.
     */
    private int inputNodeCount;

    /**
     * {@link #candidateCount} is used to check if no variable is selected. If {@link #inputNodeCount} equals
     * {@link #candidateCount}, which means no column is selected or all columns are selected.
     */
    private int candidateCount;

    /**
     * Final results map, this map is loaded in memory for sum, and will be written by context in cleanup.
     */
    private Map<Long, Double> results = new HashMap<Long, Double>();

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
     * Prevent too many new objects for output value.
     */
    private DoubleWritable outputValue;

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
            this.modelConfig = CommonUtils.loadModelConfig(
                    context.getConfiguration().get(Constants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Load first model in model path as a {@link BasicNetwork} instance.
     */
    private void loadModel() throws IOException {
        PathFinder pathFinder = new PathFinder(this.modelConfig);
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource());
        String modelSuffix = "." + this.modelConfig.getAlgorithm().toLowerCase();
        List<FileStatus> fileList = new ArrayList<FileStatus>();
        Path path = new Path(pathFinder.getModelsPath());
        fileList.addAll(Arrays.asList(fs.listStatus(path, new FileSuffixPathFilter(modelSuffix))));

        Collections.sort(fileList, new Comparator<FileStatus>() {
            @Override
            public int compare(FileStatus f1, FileStatus f2) {
                return f1.getPath().getName().compareToIgnoreCase(f2.getPath().getName());
            }
        });

        for(FileStatus f: fileList) {
            FSDataInputStream stream = null;
            try {
                stream = fs.open(f.getPath());
                this.model = BasicNetwork.class.cast(EncogDirectoryPersistence.loadObject(stream));
                break;
            } catch (RuntimeException e) {
                throw new RuntimeException("Only Neural Network so far supported in sentivity variable selection.", e);
            } finally {
                IOUtils.closeQuietly(stream);
            }
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

    /**
     * Do initialization like ModelConfig and ColumnConfig loading, model loading and others like input or output number
     * loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);
        loadModel();
        this.wrapperBy = context.getConfiguration().get(Constants.SHIFU_VARSELECT_WRAPPER_TYPE, "SE");
        int[] inputOutputIndex = getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.candidateCount = inputOutputIndex[2];
        this.inputs = new double[this.inputNodeCount];
        this.outputs = new double[inputOutputIndex[1]];
        this.columnIndexes = new long[this.inputNodeCount];
        this.inputsMLData = new BasicMLData(this.inputNodeCount);
        this.outputKey = new LongWritable();
        this.outputValue = new DoubleWritable();
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        int index = 0, inputsIndex = 0, outputsIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(value.toString())) {
            double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
            if(index == this.columnConfigList.size()) {
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig.isTarget()) {
                    this.outputs[outputsIndex++] = doubleValue;
                } else {
                    if(this.inputNodeCount == this.candidateCount) {
                        // all variables are not set final-select
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()) {
                            inputs[inputsIndex] = doubleValue;
                            columnIndexes[inputsIndex++] = columnConfig.getColumnNum();
                        }
                    } else {
                        // final select some variables
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputs[inputsIndex] = doubleValue;
                            columnIndexes[inputsIndex++] = columnConfig.getColumnNum();
                        }
                    }
                }
            }
            index++;
        }

        double oldValue = 0.0d;

        this.inputsMLData.setData(this.inputs);

        double candidateModelScore = 0d;
        if("SE".equalsIgnoreCase(this.wrapperBy)) {
            candidateModelScore = this.model.compute(new BasicMLData(inputs)).getData()[0];
        }
        for(int i = 0; i < this.inputs.length; i++) {
            oldValue = this.inputs[i];
            this.inputs[i] = 0d;
            this.inputsMLData.setData(this.inputs);
            double currentModelScore = this.model.compute(new BasicMLData(inputs)).getData()[0];

            Double MSESum = this.results.get(this.columnIndexes[i]);

            double diff = 0d;
            if("A".equalsIgnoreCase(this.wrapperBy) || "R".equalsIgnoreCase(this.wrapperBy)) {
                diff = this.outputs[0] - currentModelScore;
            } else {
                // SE
                diff = candidateModelScore - currentModelScore;
            }
            if(MSESum == null) {
                MSESum = power2(diff);
            } else {
                MSESum += power2(diff);
            }
            this.results.put(this.columnIndexes[i], MSESum);
            this.inputs[i] = oldValue;
        }

    }

    /**
     * Write all column->MSE pairs to output.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        LOG.debug("Final results: {}", results);
        for(Entry<Long, Double> entry: results.entrySet()) {
            this.outputKey.set(entry.getKey());
            // value is sumValue, not sumValue/(number of records)
            this.outputValue.set(entry.getValue());
            context.write(this.outputKey, this.outputValue);
        }
    }

    private double power2(double data) {
        return data * data;
    }

}
