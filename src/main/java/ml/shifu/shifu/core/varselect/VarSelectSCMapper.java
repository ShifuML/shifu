package ml.shifu.shifu.core.varselect;

import com.google.common.base.Splitter;
import ml.shifu.guagua.util.MemoryUtils;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.CacheBasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.CacheFlatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.*;
import org.apache.commons.collections.CollectionUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.encog.ml.MLRegression;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.persist.PersistorRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class VarSelectSCMapper extends Mapper<LongWritable, Text, LongWritable, ColumnScore> {

    private final static Logger LOG = LoggerFactory.getLogger(VarSelectSCMapper.class);

    /**
     * Model Config read from HDFS, be static to shared in multiple mappers
     */
    private static ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS, be static to shared in multiple mappers
     */
    private static List<ColumnConfig> columnConfigList;

    /**
     * Basic neural network model instance to compute basic score with all selected columns and wrapper selected
     * columns
     */
    private MLRegression model;

    /**
     * Basic input node count for NN model, all the variables selected in current model training.
     */
    private int inputNodeCount;

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
     * Filter by sensitivity by target(ST) or sensitivity(SE).
     */
    private String filterBy;

    /**
     * A counter to count # of records in current mapper.
     */
    private long recordCount;

    /**
     * Feature set used to check if column with index are in the feature set
     */
    private Set<Integer> featureSet;

    /**
     * Network which will cache first layer outputs and later use minus to replace sum to save CPU time.
     */
    private CacheBasicFloatNetwork cacheNetwork;

    /**
     * The splitter for normalization data set
     */
    private Splitter splitter;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
    private synchronized static void loadConfigFiles(final Context context) {
        if(modelConfig == null) {
            LOG.info("Before loading config with memory {} in thread {}.", MemoryUtils.getRuntimeMemoryStats(),
                    Thread.currentThread().getName());
            long start = System.currentTimeMillis();
            try {
                modelConfig = CommonUtils.loadModelConfig(Constants.MODEL_CONFIG_JSON_FILE_NAME, SourceType.LOCAL);
                columnConfigList = CommonUtils.loadColumnConfigList(Constants.COLUMN_CONFIG_JSON_FILE_NAME,
                        SourceType.LOCAL);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            LOG.info("After loading config with time {}ms and memory {} in thread {}.",
                    (System.currentTimeMillis() - start), MemoryUtils.getRuntimeMemoryStats(),
                    Thread.currentThread().getName());
        }
    }

    /**
     * Load first model in model path as a {@link MLRegression} instance.
     */
    private synchronized void loadModel() throws IOException {
        LOG.debug("Before loading model with memory {} in thread {}.", MemoryUtils.getRuntimeMemoryStats(),
                Thread.currentThread().getName());
        long start = System.currentTimeMillis();
        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(SourceType.LOCAL);

        // load model from local d-cache model file
        if(CommonUtils.isTensorFlowModel(modelConfig.getAlgorithm())) {
            this.model = (MLRegression) (ModelSpecLoaderUtils.loadBasicModels(modelConfig, null).get(0));
        } else {
            model = (MLRegression) ModelSpecLoaderUtils.loadModel(modelConfig,
                    new Path("model0." + modelConfig.getAlgorithm().toLowerCase()), fs);
        }
        LOG.debug("After load model class {} with time {}ms and memory {} in thread {}.", model.getClass().getName(),
                (System.currentTimeMillis() - start), MemoryUtils.getRuntimeMemoryStats(),
                Thread.currentThread().getName());
    }

    /**
     * Copy existing model to {@link CacheBasicFloatNetwork} model for fast scoring in sensitivity computing.
     *
     * @param network
     *            the raw network model
     * @return the cache network model instance.
     */
    public static final CacheBasicFloatNetwork copy(final BasicFloatNetwork network) {
        final CacheBasicFloatNetwork result = new CacheBasicFloatNetwork(network);
        final CacheFlatNetwork flat = new CacheFlatNetwork();
        result.getProperties().putAll(network.getProperties());

        flat.setBeginTraining(network.getFlat().getBeginTraining());
        flat.setConnectionLimit(network.getFlat().getConnectionLimit());
        flat.setContextTargetOffset(network.getFlat().getContextTargetOffset());
        flat.setContextTargetSize(network.getFlat().getContextTargetSize());
        flat.setEndTraining(network.getFlat().getEndTraining());
        flat.setHasContext(network.getFlat().getHasContext());
        flat.setInputCount(network.getFlat().getInputCount());
        flat.setLayerCounts(network.getFlat().getLayerCounts());
        flat.setLayerFeedCounts(network.getFlat().getLayerFeedCounts());
        flat.setLayerContextCount(network.getFlat().getLayerContextCount());
        flat.setLayerIndex(network.getFlat().getLayerIndex());
        flat.setLayerOutput(network.getFlat().getLayerOutput());
        flat.setLayerSums(network.getFlat().getLayerSums());
        flat.setOutputCount(network.getFlat().getOutputCount());
        flat.setWeightIndex(network.getFlat().getWeightIndex());
        flat.setWeights(network.getFlat().getWeights());
        flat.setBiasActivation(network.getFlat().getBiasActivation());
        flat.setActivationFunctions(network.getFlat().getActivationFunctions());
        result.setFeatureSet(network.getFeatureSet());
        result.getStructure().setFlat(flat);
        return result;
    }

    /**
     * Do initialization like ModelConfig and ColumnConfig loading, model loading and others like input or output number
     * loading.
     */
    @Override
    protected void setup(Context context) throws IOException {
        loadConfigFiles(context);

        loadModel();

        // Copy mode to here
        cacheNetwork = copy((BasicFloatNetwork) model);

        this.filterBy = context.getConfiguration().get(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE,
                Constants.FILTER_BY_SE);
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        if(model instanceof BasicFloatNetwork) {
            this.inputs = new double[((BasicFloatNetwork) model).getFeatureSet().size()];
            this.featureSet = ((BasicFloatNetwork) model).getFeatureSet();
        } else {
            this.inputs = new double[this.inputNodeCount];
        }

        boolean isAfterVarSelect = (inputOutputIndex[0] != 0);
        // cache all feature list for sampling features
        if(CollectionUtils.isEmpty(this.featureSet)) {
            this.featureSet = new HashSet<Integer>(NormalUtils.getAllFeatureList(columnConfigList, isAfterVarSelect));
            this.inputs = new double[this.featureSet.size()];
        }

        if(inputs.length != this.inputNodeCount) {
            throw new IllegalArgumentException("Model input count " + model.getInputCount()
                    + " is inconsistent with input size " + this.inputNodeCount + ".");
        }

        this.outputs = new double[inputOutputIndex[1]];
        this.columnIndexes = new long[this.inputs.length];
        this.inputsMLData = new BasicMLData(this.inputs.length);
        LOG.info("Filter by is {}", filterBy);

        // create Splitter
        String delimiter = context.getConfiguration().get(Constants.SHIFU_OUTPUT_DATA_DELIMITER);
        this.splitter = MapReduceUtils.generateShifuOutputSplitter(delimiter);
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        recordCount += 1L;
        double weight = 1.0d;
        int index = 0, inputsIndex = 0, outputsIndex = 0;
        for(String input: this.splitter.split(value.toString())) {
            double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
            if(index == columnConfigList.size()) {
                // last column of the input, it's weight
                weight = doubleValue;
                break;
            } else {
                ColumnConfig columnConfig = columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    this.outputs[outputsIndex++] = doubleValue;
                } else {
                    if(this.featureSet != null && this.featureSet.contains(columnConfig.getColumnNum())) {
                        inputs[inputsIndex] = doubleValue;
                        columnIndexes[inputsIndex++] = columnConfig.getColumnNum();
                    }
                }
            }
            index++;
        }

        this.inputsMLData.setData(this.inputs);
        // compute candidate model score , cache first layer of sum values in such call method, cache flag here is true
        double candidateModelScore = cacheNetwork.compute(inputsMLData, true, -1).getData()[0];
        // output the actual score as column id -1, then user could know the actual catch rate
        // before dropping any variables
        ColumnScore actualScore = new ColumnScore();
        actualScore.setColumnTag((int) this.outputs[0]);
        actualScore.setWeight(weight);
        actualScore.setSensitivityScore(candidateModelScore);
        context.write(new LongWritable(-1), actualScore);

        for(int i = 0; i < this.inputs.length; i++) {
            // cache flag is false to reuse cache sum of first layer of values.
            double currentModelScore = cacheNetwork.compute(inputsMLData, false, i).getData()[0];
            ColumnScore columnScore = new ColumnScore();
            columnScore.setColumnTag((int) this.outputs[0]);
            columnScore.setWeight(weight);
            columnScore.setSensitivityScore(currentModelScore);
            context.write(new LongWritable(this.columnIndexes[i]), columnScore);
            if(this.recordCount % 1000 == 0) {
                LOG.info("The column score for {} is {}", this.columnIndexes[i], columnScore);
            }
        }

        if(this.recordCount % 1000 == 0) {
            LOG.info("Finish to process {} records.", this.recordCount);
        }

    }

}
