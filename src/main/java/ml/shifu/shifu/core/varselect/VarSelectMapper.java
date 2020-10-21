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

import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import ml.shifu.guagua.util.MemoryUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.CacheBasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.CacheFlatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.udf.norm.CategoryMissingNormType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.MapReduceUtils;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;
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
import java.util.*;
import java.util.Map.Entry;

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
    @SuppressWarnings("unused")
    private static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR);

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
     * Network which will cache first layer outputs and later use minus to replace sum to save CPU time.
     */
    private CacheBasicFloatNetwork cacheNetwork;

    /**
     * Final results map, this map is loaded in memory for sum, and will be written by context in cleanup.
     */
    private Map<Long, ColumnInfo> results = new HashMap<>();

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

    private int featureInputLen;

    /**
     * Feature set used to check if column with index are in the feature set
     */
    private Set<Integer> featureSet;

    /**
     * The input values for each columns, if the input is missing
     */
    private Map<Integer, List<Double>> columnMissingInputValues;

    /**
     * Mapping the column id to offset of normalization data
     */
    private Map<Integer, Integer> columnNormDataPosMapping;

    /**
     * The splitter for normalization data set
     */
    private Splitter splitter;

    private boolean isLinearTarget;

    private boolean hasCandidates;

    /**
     * Do initialization like ModelConfig and ColumnConfig loading, model loading and others like input or output number
     * loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        loadModel();

        // Copy mode to here
        if(CommonUtils.isTensorFlowModel(modelConfig.getAlgorithm())) {
            cacheNetwork = null;
        } else {
            cacheNetwork = copy((BasicFloatNetwork) model);
        }

        this.filterBy = context.getConfiguration().get(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE,
                Constants.FILTER_BY_SE);
        LOG.info("Filter by is {}", filterBy);

        this.featureSet = DTrainUtils.generateModelFeatureSet(modelConfig, columnConfigList);
        this.columnMissingInputValues = new HashMap<>();
        this.columnNormDataPosMapping = new HashMap<>();
        this.featureInputLen = DTrainUtils.generateFeatureInputInfo(modelConfig, columnConfigList, this.featureSet,
                this.columnMissingInputValues, this.columnNormDataPosMapping);

        int modelInputCount = model.getInputCount();
        Set<Integer> modelFeatureSet = null;
        if(model instanceof BasicFloatNetwork) {
            modelFeatureSet = ((BasicFloatNetwork) model).getFeatureSet();
        }

        if (CollectionUtils.isEmpty(featureSet) || this.featureInputLen == 0) {
            throw new IllegalArgumentException("No input columns according to ColumnConfig.json. Please check!");
        }

        if(this.featureInputLen != modelInputCount) {
            throw new IllegalArgumentException("Model input count " + modelInputCount
                    + " is inconsistent with input size " + this.featureInputLen + ".");
        }

        if(CollectionUtils.isNotEmpty(modelFeatureSet)
                && (this.featureSet.size() != modelFeatureSet.size() || !this.featureSet.containsAll(modelFeatureSet))){
            // both feature set and model feature set are not empty
            throw new IllegalArgumentException("Model input features is inconsistent with ColumnConfig.json. "
                    + "Please check the model spec vs. ColumnConfig.json");
        }

        this.inputs = new double[this.featureInputLen];
        this.outputs = new double[this.model.getOutputCount()];

        this.inputsMLData = new BasicMLData(this.inputs.length);
        this.outputKey = new LongWritable();

        // create Splitter
        String delimiter = context.getConfiguration().get(Constants.SHIFU_OUTPUT_DATA_DELIMITER);
        this.splitter = MapReduceUtils.generateShifuOutputSplitter(delimiter);

        this.isLinearTarget = CommonUtils.isLinearTarget(modelConfig, columnConfigList);
        this.hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
    }

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
        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(SourceType.LOCAL, null);
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

    @SuppressWarnings("unused")
    private Set<Integer> generateModelFeatureSet(List<ColumnConfig> columnConfigList) {
        Set<Integer> columnIdSet = new HashSet<>();
        boolean hasFinalSelectedVars = DTrainUtils.hasFinalSelectedVars(columnConfigList);
        if (hasFinalSelectedVars) {
            columnConfigList.stream().forEach(columnConfig -> {
                if (columnConfig.isFinalSelect()) {
                    columnIdSet.add(columnConfig.getColumnNum());
                }
            });
        } else {
            boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
            columnConfigList.stream().forEach(columnConfig -> {
                if (CommonUtils.isGoodCandidate(columnConfig, hasCandidates, modelConfig.isRegression())) {
                    columnIdSet.add(columnConfig.getColumnNum());
                }
            });
        }
        return columnIdSet;
    }

    @SuppressWarnings("unused")
    private int generateFeatureInputInfo(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Set<Integer> featureSet) {
        int vectorLen = 0;
        this.columnMissingInputValues = new HashMap<>();
        this.columnNormDataPosMapping = new HashMap<>();
        for ( ColumnConfig columnConfig : columnConfigList) {
            if (featureSet.contains(columnConfig.getColumnNum())) {
                this.columnNormDataPosMapping.put(columnConfig.getColumnNum(), vectorLen);
                List<Double> normValues = Normalizer.normalize(columnConfig, null,
                        modelConfig.getNormalizeStdDevCutOff(), modelConfig.getNormalizeType(),
                        CategoryMissingNormType.MEAN);
                if(CollectionUtils.isNotEmpty(normValues)) { // usually, the normValues won't be empty
                    this.columnMissingInputValues.put(columnConfig.getColumnNum(), normValues);
                    vectorLen += normValues.size();
                }
            }
        }
        return vectorLen;
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        recordCount += 1L;

        if(recordCount % 200 == 0) {
            LOG.info("Count {} with Memory used by {}.", recordCount, MemoryUtils.getRuntimeMemoryStats());
        }

        // load normalized data into inputs
        DTrainUtils.loadDataIntoInputs(modelConfig, columnConfigList, this.featureSet,
                this.isLinearTarget, this.hasCandidates, inputs, outputs,
                Lists.newArrayList(this.splitter.split(value.toString())).toArray(new String[0]));

        // set inputs into inputsMLData
        this.inputsMLData.setData(this.inputs);
        // compute candidate model score , cache first layer of sum values in such call method, cache flag here is true
        double candidateModelScore;
        if(CommonUtils.isTensorFlowModel(modelConfig.getAlgorithm())) {
            candidateModelScore = this.model.compute(inputsMLData).getData(0);
        } else {
            candidateModelScore = cacheNetwork.compute(inputsMLData, true, -1).getData()[0];
        }

        for (ColumnConfig columnConfig: columnConfigList) {
            if (this.featureSet.contains(columnConfig.getColumnNum())) {
                List<Double> missingVals = this.columnMissingInputValues.get(columnConfig.getColumnNum());
                int startOps = this.columnNormDataPosMapping.get(columnConfig.getColumnNum());
                double currentModelScore;
                if(CommonUtils.isTensorFlowModel(modelConfig.getAlgorithm())) {
                    double[] newInputs = Arrays.copyOf(inputsMLData.getData(), inputsMLData.getData().length);
                    for (int i = 0; i < missingVals.size(); i ++) {
                        newInputs[startOps + i] = missingVals.get(i);
                    }
                    currentModelScore = this.model.compute(new BasicMLData(newInputs)).getData(0);
                } else {
                    currentModelScore = cacheNetwork.compute(inputsMLData, false, startOps).getData()[0];
                }

                double diff = 0d;
                if(Constants.FILTER_BY_ST.equalsIgnoreCase(this.filterBy)) { // ST
                    diff = this.outputs[0] - currentModelScore;
                } else { // SE
                    diff = candidateModelScore - currentModelScore;
                }

                ColumnInfo columnInfo = this.results.get(columnConfig.getColumnNum().longValue());
                if(columnInfo == null) {
                    columnInfo = new ColumnInfo();
                    columnInfo.setSumScoreDiff(Math.abs(diff));
                    columnInfo.setSumSquareScoreDiff(power2(diff));
                    this.results.put(columnConfig.getColumnNum().longValue(), columnInfo);
                } else {
                    columnInfo.setSumScoreDiff(columnInfo.getSumScoreDiff() + Math.abs(diff));
                    columnInfo.setSumSquareScoreDiff(columnInfo.getSumSquareScoreDiff() + power2(diff));
                }
            }
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
