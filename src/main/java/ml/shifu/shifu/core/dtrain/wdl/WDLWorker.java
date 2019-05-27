/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wdl;

import com.google.common.base.Splitter;
import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.MapReduceUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * {@link WDLWorker} is responsible for loading part of data into memory, do iteration gradients computation and send
 * back to master for master aggregation. After master aggregation is done, received latest weights to do next
 * iteration.
 * 
 * <p>
 * Data loading into memory as memory list includes two parts: numerical float array and sparse input object array which
 * is for categorical variables. To leverage sparse feature of categorical variables, sparse object is leveraged to
 * save memory and matrix computation.
 * 
 * <p>
 * First iteration, just return empty to master but wait for next iteration master models sync-up. Since at very first
 * model training in all workers should be starting from the same model.
 * 
 * <p>
 * After {@link #wnd} updating weights from master result each iteration. Then do forward-backward computation and in
 * backward computation of each record to compute gradients. Such gradients arch as wide and deep graph needs to be
 * aggregated and sent back to master.
 * 
 * <p>
 * TODO mini batch matrix support, matrix computation support
 * TODO variable/field based optimization to compute gradients
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 3600)
public class WDLWorker extends
        AbstractWorkerComputable<WDLParams, WDLParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(WDLWorker.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Basic input count for final-select variables or good candidates(if no any variables are selected)
     */
    protected int inputCount;

    /**
     * Basic numerical input count for final-select variables or good candidates(if no any variables are selected)
     */
    protected int numInputs;

    /**
     * Basic categorical input count
     */
    protected int cateInputs;

    /**
     * Means if do variable selection, if done, many variables will be set to finalSelect = true; if not, no variables
     * are selected and should be set to all good candidate variables.
     */
    private boolean isAfterVarSelect = true;

    /**
     * input record size, inc one by one.
     */
    protected long count;

    /**
     * sampled input record size.
     */
    protected long sampleCount;

    /**
     * Positive count in training data list, only be effective in 0-1 regression or onevsall classification
     */
    protected long positiveTrainCount;

    /**
     * Positive count in training data list and being selected in training, only be effective in 0-1 regression or
     * onevsall classification
     */
    protected long positiveSelectedTrainCount;

    /**
     * Negative count in training data list , only be effective in 0-1 regression or onevsall classification
     */
    protected long negativeTrainCount;

    /**
     * Negative count in training data list and being selected, only be effective in 0-1 regression or onevsall
     * classification
     */
    protected long negativeSelectedTrainCount;

    /**
     * Positive count in validation data list, only be effective in 0-1 regression or onevsall classification
     */
    protected long positiveValidationCount;

    /**
     * Negative count in validation data list, only be effective in 0-1 regression or onevsall classification
     */
    protected long negativeValidationCount;

    /**
     * Training data set with only in memory, in the future, MemoryDiskList can be leveraged.
     */
    private volatile MemoryLimitedList<Data> trainingData;

    /**
     * Validation data set with only in memory.
     */
    private volatile MemoryLimitedList<Data> validationData;

    /**
     * Mapping for (ColumnNum, Map(Category, CategoryIndex) for categorical feature
     */
    private Map<Integer, Map<String, Integer>> columnCategoryIndexMapping;

    /**
     * A splitter to split data with specified delimiter.
     */
    private Splitter splitter;

    /**
     * Index map in which column index and data input array index for fast location.
     */
    private ConcurrentMap<Integer, Integer> inputIndexMap = new ConcurrentHashMap<Integer, Integer>();

    /**
     * Trainer id used to tag bagging training job, starting from 0, 1, 2 ...
     */
    private int trainerId = 0;

    /**
     * If has candidate in column list.
     */
    private boolean hasCandidates;

    /**
     * Indicates if validation are set by users for validationDataPath, not random picking
     */
    protected boolean isManualValidation = false;

    /**
     * If stratified sampling or random sampling
     */
    private boolean isStratifiedSampling = false;

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;

    /**
     * Construct a validation random map for different classes. For stratified sampling, this is useful for class level
     * sampling.
     */
    private Map<Integer, Random> validationRandomMap = new HashMap<Integer, Random>();

    /**
     * Random object to sample negative records
     */
    protected Random negOnlyRnd = new Random(System.currentTimeMillis() + 1000L);

    /**
     * Whether to enable poisson bagging with replacement.
     */
    protected boolean poissonSampler;

    /**
     * PoissonDistribution which is used for poisson sampling for bagging with replacement.
     */
    protected PoissonDistribution rng = null;

    /**
     * PoissonDistribution which is used for up sampling positive records.
     */
    protected PoissonDistribution upSampleRng = null;

    /**
     * Parameters defined in ModelConfig.json#train part
     */
    private Map<String, Object> validParams;

    /**
     * WideAndDeep graph definition network.
     */
    private WideAndDeep wnd;

    /**
     * Logic to load data into memory list which includes float array for numerical features and sparse object array for
     * categorical features.
     */
    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<WDLParams, WDLParams> context) {
        if((++this.count) % 5000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        // hashcode for fixed input split in train and validation
        long hashcode = 0;
        float[] inputs = new float[this.numInputs];
        this.cateInputs = (int) this.columnConfigList.stream().filter(ColumnConfig::isCategorical).count();
        SparseInput[] cateInputs = new SparseInput[this.cateInputs];
        float ideal = 0f, significance = 1f;
        int index = 0, numIndex = 0, cateIndex = 0;
        // use guava Splitter to iterate only once
        for(String input: this.splitter.split(currentValue.getWritable().toString())) {
            // if no wgt column at last pos, no need process here 
            if(index == this.columnConfigList.size()) {
                significance = getWeightValue(input);
                break; // the last field is significance, break here
            } else {
                ColumnConfig config = this.columnConfigList.get(index);
                if(config != null && config.isTarget()) {
                    ideal = getFloatValue(input);
                } else {
                    // final select some variables but meta and target are not included
                    if(validColumn(config)) {
                        if(config.isNumerical()) {
                            inputs[numIndex] = getFloatValue(input);
                            this.inputIndexMap.putIfAbsent(config.getColumnNum(), numIndex++);
                        } else if(config.isCategorical()) {
                            cateInputs[cateIndex] = new SparseInput(config.getColumnNum(), getCateIndex(input, config));
                            this.inputIndexMap.putIfAbsent(config.getColumnNum(), cateIndex++);
                        }
                        hashcode = hashcode * 31 + input.hashCode();
                    }
                }
            }
            index += 1;
        }

        // output delimiter in norm can be set by user now and if user set a special one later changed, this exception
        // is helped to quick find such issue.
        validateInputLength(context, inputs, numIndex);

        // sample negative only logic here
        if(sampleNegOnly(hashcode, ideal)) {
            return;
        }
        // up sampling logic, just add more weights while bagging sampling rate is still not changed
        if(modelConfig.isRegression() && isUpSampleEnabled() && Double.compare(ideal, 1d) == 0) {
            // ideal == 1 means positive tags; sample + 1 to avoid sample count to 0
            significance = significance * (this.upSampleRng.sample() + 1);
        }

        Data data = new Data(inputs, cateInputs, significance, ideal);
        // split into validation and training data set according to validation rate
        boolean isInTraining = this.addDataPairToDataSet(hashcode, data, context.getAttachment());
        // update some positive or negative selected count in metrics
        this.updateMetrics(data, isInTraining);
    }

    protected boolean isUpSampleEnabled() {
        // only enabled in regression
        return this.upSampleRng != null && (modelConfig.isRegression()
                || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()));
    }

    private boolean sampleNegOnly(long hashcode, float ideal) {
        boolean ret = false;
        if(modelConfig.getTrain().getSampleNegOnly()) {
            double bagSampleRate = this.modelConfig.getBaggingSampleRate();
            if(this.modelConfig.isFixInitialInput()) {
                // if fixInitialInput, sample hashcode in 1-sampleRate range out if negative records
                int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                // here BaggingSampleRate means how many data will be used in training and validation, if it is 0.8, we
                // should take 1-0.8 to check endHashCode
                int endHashCode = startHashCode + Double.valueOf((1d - bagSampleRate) * 100).intValue();
                if((modelConfig.isRegression()
                        || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()))
                        && (int) (ideal + 0.01d) == 0 && isInRange(hashcode, startHashCode, endHashCode)) {
                    ret = true;
                }
            } else {
                // if not fixed initial input, for regression or onevsall multiple classification, if negative record
                if((modelConfig.isRegression()
                        || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()))
                        && (int) (ideal + 0.01d) == 0 && negOnlyRnd.nextDouble() > bagSampleRate) {
                    ret = true;
                }
            }
        }
        return ret;
    }

    private void updateMetrics(Data data, boolean isInTraining) {
        // do bagging sampling only for training data
        if(isInTraining) {
            // for training data, compute real selected training data according to baggingSampleRate
            if(isPositive(data.label)) {
                this.positiveSelectedTrainCount += 1L;
            } else {
                this.negativeSelectedTrainCount += 1L;
            }
        } else {
            // for validation data, according bagging sampling logic, we may need to sampling validation data set, while
            // validation data set are only used to compute validation error, not to do real sampling is ok.
        }
    }

    /**
     * Add to training set or validation set according to validation rate.
     * 
     * @param hashcode
     *            the hash code of the data
     * @param data
     *            data instance
     * @param attachment
     *            if it is validation
     * @return if in training, training is true, others are false.
     */
    protected boolean addDataPairToDataSet(long hashcode, Data data, Object attachment) {
        // if validation data from configured validation data set
        boolean isValidation = (attachment != null && attachment instanceof Boolean) ? (Boolean) attachment : false;

        if(this.isKFoldCV) {
            int k = this.modelConfig.getTrain().getNumKFold();
            if(hashcode % k == this.trainerId) {
                this.validationData.append(data);
                if(isPositive(data.label)) {
                    this.positiveValidationCount += 1L;
                } else {
                    this.negativeValidationCount += 1L;
                }
                return false;
            } else {
                this.trainingData.append(data);
                if(isPositive(data.label)) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        }

        if(this.isManualValidation) {
            if(isValidation) {
                this.validationData.append(data);
                if(isPositive(data.label)) {
                    this.positiveValidationCount += 1L;
                } else {
                    this.negativeValidationCount += 1L;
                }
                return false;
            } else {
                this.trainingData.append(data);
                if(isPositive(data.label)) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        } else {
            if(Double.compare(this.modelConfig.getValidSetRate(), 0d) != 0) {
                int classValue = (int) (data.label + 0.01f);
                Random random = null;
                if(this.isStratifiedSampling) {
                    // each class use one random instance
                    random = validationRandomMap.get(classValue);
                    if(random == null) {
                        random = new Random();
                        this.validationRandomMap.put(classValue, random);
                    }
                } else {
                    // all data use one random instance
                    random = validationRandomMap.get(0);
                    if(random == null) {
                        random = new Random();
                        this.validationRandomMap.put(0, random);
                    }
                }

                if(this.modelConfig.isFixInitialInput()) {
                    // for fix initial input, if hashcode%100 is in [start-hashcode, end-hashcode), validation,
                    // otherwise training. start hashcode in different job is different to make sure bagging jobs have
                    // different data. if end-hashcode is over 100, then check if hashcode is in [start-hashcode, 100]
                    // or [0, end-hashcode]
                    int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                    int endHashCode = startHashCode
                            + Double.valueOf(this.modelConfig.getValidSetRate() * 100).intValue();
                    if(isInRange(hashcode, startHashCode, endHashCode)) {
                        this.validationData.append(data);
                        if(isPositive(data.label)) {
                            this.positiveValidationCount += 1L;
                        } else {
                            this.negativeValidationCount += 1L;
                        }
                        return false;
                    } else {
                        this.trainingData.append(data);
                        if(isPositive(data.label)) {
                            this.positiveTrainCount += 1L;
                        } else {
                            this.negativeTrainCount += 1L;
                        }
                        return true;
                    }
                } else {
                    // not fixed initial input, if random value >= validRate, training, otherwise validation.
                    if(random.nextDouble() >= this.modelConfig.getValidSetRate()) {
                        this.trainingData.append(data);
                        if(isPositive(data.label)) {
                            this.positiveTrainCount += 1L;
                        } else {
                            this.negativeTrainCount += 1L;
                        }
                        return true;
                    } else {
                        this.validationData.append(data);
                        if(isPositive(data.label)) {
                            this.positiveValidationCount += 1L;
                        } else {
                            this.negativeValidationCount += 1L;
                        }
                        return false;
                    }
                }
            } else {
                this.trainingData.append(data);
                if(isPositive(data.label)) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        }
    }

    private boolean isPositive(float value) {
        return Float.compare(1f, value) == 0;
    }

    private boolean isInRange(long hashcode, int startHashCode, int endHashCode) {
        // check if in [start, end] or if in [start, 100) and [0, end-100)
        int hashCodeIn100 = (int) hashcode % 100;
        if(endHashCode <= 100) {
            // in range [start, end)
            return hashCodeIn100 >= startHashCode && hashCodeIn100 < endHashCode;
        } else {
            // in range [start, 100) or [0, endHashCode-100)
            return hashCodeIn100 >= startHashCode || hashCodeIn100 < (endHashCode % 100);
        }
    }

    /**
     * If no enough columns for model training, most of the cases root cause is from inconsistent delimiter.
     */
    private void validateInputLength(WorkerContext<WDLParams, WDLParams> context, float[] inputs, int numInputIndex) {
        if(numInputIndex != inputs.length) {
            String delimiter = context.getProps().getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER,
                    Constants.DEFAULT_DELIMITER);
            throw new RuntimeException("Input length is inconsistent with parsing size. Input original size: "
                    + inputs.length + ", parsing size:" + numInputIndex + ", delimiter:" + delimiter + ".");
        }
    }

    /**
     * If column is valid and be selected in model training
     */
    private boolean validColumn(ColumnConfig columnConfig) {
        if(isAfterVarSelect) {
            return columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                    && columnConfig.isFinalSelect();
        } else {
            return !columnConfig.isMeta() && !columnConfig.isTarget()
                    && CommonUtils.isGoodCandidate(columnConfig, this.hasCandidates);
        }
    }

    private int getCateIndex(String input, ColumnConfig columnConfig) {
        int shortValue = (columnConfig.getBinCategory().size());
        if(input.length() == 0) { // missing which is invalid category
            shortValue = columnConfig.getBinCategory().size();
        } else {
            Integer cateIndex = this.columnCategoryIndexMapping.get(columnConfig.getColumnNum()).get(input);
            shortValue = (cateIndex == null) ? -1 : cateIndex.intValue(); // -1 is invalid or not existing category
            if(shortValue == -1) { // still not found
                shortValue = columnConfig.getBinCategory().size();
            }
        }
        return shortValue;
    }

    private float getWeightValue(String input) {
        float significance = 1f;
        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())) {
            // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 1f)
            significance = input.length() == 0 ? 1f : NumberFormatUtils.getFloat(input, 1f);
            // if invalid weight, set it to 1f and warning in log
            if(significance < 0f) {
                LOG.warn("Record {} with weight {} is less than 0 and invalid, set it to 1.", count, significance);
                significance = 1f;
            }
        }
        return significance;
    }

    private float getFloatValue(String input) {
        // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
        float floatValue = input.length() == 0 ? 0f : NumberFormatUtils.getFloat(input, 0f);
        // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
        return (Float.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;
    }

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        // initialize Hadoop based line (long, string) reader
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @SuppressWarnings({ "unchecked", "unused" })
    @Override
    public void init(WorkerContext<WDLParams, WDLParams> context) {
        Properties props = context.getProps();
        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.initCateIndexMap();
        this.hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        // create Splitter
        String delimiter = context.getProps().getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER);
        this.splitter = MapReduceUtils.generateShifuOutputSplitter(delimiter);

        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
            LOG.info("Cross validation is enabled by kCrossValidation: {}.", kCrossValidation);
        }

        this.poissonSampler = Boolean.TRUE.toString()
                .equalsIgnoreCase(context.getProps().getProperty(NNConstants.NN_POISON_SAMPLER));
        this.rng = new PoissonDistribution(1d);
        Double upSampleWeight = modelConfig.getTrain().getUpSampleWeight();
        if(upSampleWeight != 1d && (modelConfig.isRegression()
                || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()))) {
            // set mean to upSampleWeight -1 and get sample + 1to make sure no zero sample value
            LOG.info("Enable up sampling with weight {}.", upSampleWeight);
            this.upSampleRng = new PoissonDistribution(upSampleWeight - 1);
        }

        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));

        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);

        double validationRate = this.modelConfig.getValidSetRate();
        if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
            // fixed 0.6 and 0.4 of max memory for trainingData and validationData
            this.trainingData = new MemoryLimitedList<Data>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * 0.6), new ArrayList<Data>());
            this.validationData = new MemoryLimitedList<Data>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * 0.4), new ArrayList<Data>());
        } else {
            if(validationRate != 0d) {
                this.trainingData = new MemoryLimitedList<Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction * (1 - validationRate)),
                        new ArrayList<Data>());
                this.validationData = new MemoryLimitedList<Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction * validationRate),
                        new ArrayList<Data>());
            } else {
                this.trainingData = new MemoryLimitedList<Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction), new ArrayList<Data>());
            }
        }

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        // numerical + categorical = # of all input
        this.numInputs = inputOutputIndex[0];
        this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];
        // regression outputNodeCount is 1, binaryClassfication, it is 1, OneVsAll it is 1, Native classification it is
        // 1, with index of 0,1,2,3 denotes different classes
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        this.isManualValidation = (modelConfig.getValidationDataSetRawPath() != null
                && !"".equals(modelConfig.getValidationDataSetRawPath()));

        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();

        this.validParams = this.modelConfig.getTrain().getParams();

        // Build wide and deep graph
        List<Integer> embedColumnIds = (List<Integer>) this.validParams.get(CommonConstants.NUM_EMBED_COLUMN_IDS);
        Integer embedOutputs = (Integer) this.validParams.get(CommonConstants.NUM_EMBED_OUTPUTS);
        List<Integer> embedOutputList = new ArrayList<Integer>();
        for(Integer cId: embedColumnIds) {
            embedOutputList.add(embedOutputs == null ? CommonConstants.DEFAULT_EMBEDING_OUTPUT : embedOutputs);
        }
        List<Integer> numericalIds = DTrainUtils.getNumericalIds(this.columnConfigList, isAfterVarSelect);
        List<Integer> wideColumnIds = DTrainUtils.getCategoricalIds(columnConfigList, isAfterVarSelect);
        Map<Integer, Integer> idBinCateSizeMap = DTrainUtils.getIdBinCategorySizeMap(columnConfigList);
        int numLayers = (Integer) this.validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        Float l2reg = ((Double) this.validParams.get(CommonConstants.WDL_L2_REG)).floatValue();
        this.wnd = new WideAndDeep(idBinCateSizeMap, numInputs, numericalIds, embedColumnIds, embedOutputList,
                wideColumnIds, hiddenNodes, actFunc, l2reg);
    }

    private void initCateIndexMap() {
        this.columnCategoryIndexMapping = new HashMap<Integer, Map<String, Integer>>();
        for(ColumnConfig config: this.columnConfigList) {
            if(config.isCategorical() && config.getBinCategory() != null) {
                Map<String, Integer> tmpMap = new HashMap<String, Integer>();
                for(int i = 0; i < config.getBinCategory().size(); i++) {
                    List<String> catVals = CommonUtils.flattenCatValGrp(config.getBinCategory().get(i));
                    for(String cval: catVals) {
                        tmpMap.put(cval, i);
                    }
                }
                this.columnCategoryIndexMapping.put(config.getColumnNum(), tmpMap);
            }
        }
    }

    @Override
    public WDLParams doCompute(WorkerContext<WDLParams, WDLParams> context) {
        if(context.isFirstIteration()) {
            // return empty which has been ignored in master first iteration, worker needs sync with master at first.
            return new WDLParams();
        }

        // update master global model into worker WideAndDeep graph
        this.wnd.updateWeights(context.getLastMasterResult());

        // forward and backward compute gradients for each iteration
        int trainCnt = trainingData.size(), validCnt = validationData.size();
        double trainSumError = 0d, validSumError = 0d;
        for(Data data: trainingData) {
            float[] logits = this.wnd.forward(data.getNumericalValues(), getEmbedInputs(data), getWideInputs(data));
            float predict = sigmoid(logits[0]);
            float error = predict - data.label;
            // TODO, logloss, squredloss, weighted error or not
            trainSumError +=  data.getWeight() * error * error;
            this.wnd.backward(new float[] { predict }, new float[] { data.label }, data.getWeight());
        }

        // compute validation error
        for(Data data: validationData) {
            float[] logits = this.wnd.forward(data.getNumericalValues(), getEmbedInputs(data), getWideInputs(data));
            float error = sigmoid(logits[0]) - data.label;
            validSumError += data.weight * error * error;
        }
        
        LOG.info("training error is {} {}", trainSumError, validSumError);
        // set cnt, error to params and return to master
        WDLParams params = new WDLParams();
        params.setTrainCount(trainCnt);
        params.setValidationCount(validCnt);
        params.setTrainError(trainSumError);
        params.setValidationError(validSumError);
        params.setSerializationType(SerializationType.GRADIENTS);
        this.wnd.setSerializationType(SerializationType.GRADIENTS);
        params.setWnd(this.wnd);
        return params;
    }

    public float sigmoid(float logit) {
        return (float) (1 / (1 + Math.min(1.0E19, Math.exp(-logit))));
    }

    private List<SparseInput> getWideInputs(Data data) {
        return this.wnd.getWideColumnIds().stream().map(id -> data.getCategoricalValues()[this.inputIndexMap.get(id)])
                .collect(Collectors.toList());
    }

    private List<SparseInput> getEmbedInputs(Data data) {
        return this.wnd.getEmbedColumnIds().stream().map(id -> data.getCategoricalValues()[this.inputIndexMap.get(id)])
                .collect(Collectors.toList());
    }

    @Override
    protected void postLoad(WorkerContext<WDLParams, WDLParams> context) {
        this.trainingData.switchState();
        if(validationData != null) {
            this.validationData.switchState();
        }
        LOG.info("    - # Records of the Total Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        if(this.isKFoldCV) {
            LOG.info("        - Validation Rate(kFold): {}.", 1d / this.modelConfig.getTrain().getNumKFold());
        } else {
            LOG.info("        - Validation Rate: {}.", this.modelConfig.getValidSetRate());
        }
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.size());
        if(modelConfig.isRegression() || modelConfig.getTrain().isOneVsAll()) {
            LOG.info("        - # Positive Bagging Selected Records of the Training Set: {}.",
                    this.positiveSelectedTrainCount);
            LOG.info("        - # Negative Bagging Selected Records of the Training Set: {}.",
                    this.negativeSelectedTrainCount);
            LOG.info("        - # Positive Raw Records of the Training Set: {}.", this.positiveTrainCount);
            LOG.info("        - # Negative Raw Records of the Training Set: {}.", this.negativeTrainCount);
        }

        if(validationData != null) {
            LOG.info("        - # Records of the Validation Set: {}.", this.validationData.size());
            if(modelConfig.isRegression() || modelConfig.getTrain().isOneVsAll()) {
                LOG.info("        - # Positive Records of the Validation Set: {}.", this.positiveValidationCount);
                LOG.info("        - # Negative Records of the Validation Set: {}.", this.negativeValidationCount);
            }
        }

    }

    /**
     * {@link Data} denotes training record with a float array of dense (numerical) inputs and a list of sparse inputs
     * of categorical input features.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    public static class Data {

        /**
         * Numerical values
         */
        private float[] numericalValues;

        /**
         * Categorical values in sparse object
         */
        private SparseInput[] categoricalValues;

        /**
         * The weight of one training record like dollar amount in one txn
         */
        private float weight;

        /**
         * Target value of one record
         */
        private float label;

        /**
         * Constructor for a unified data object which is for a line of training record.
         * 
         * @param numericalValues
         *            numerical values
         * @param categoricalValues
         *            categorical values which stored into one {@link SparseInput} array.
         * @param weight
         *            the weight of one training record
         * @param ideal
         *            the label field, 0 or 1
         */
        public Data(float[] numericalValues, SparseInput[] categoricalValues, float weight, float ideal) {
            this.numericalValues = numericalValues;
            this.categoricalValues = categoricalValues;
            this.weight = weight;
            this.label = ideal;
        }

        /**
         * @return the numericalValues
         */
        public float[] getNumericalValues() {
            return numericalValues;
        }

        /**
         * @param numericalValues
         *            the numericalValues to set
         */
        public void setNumericalValues(float[] numericalValues) {
            this.numericalValues = numericalValues;
        }

        /**
         * @return the categoricalValues
         */
        public SparseInput[] getCategoricalValues() {
            return categoricalValues;
        }

        /**
         * @param categoricalValues
         *            the categoricalValues to set
         */
        public void setCategoricalValues(SparseInput[] categoricalValues) {
            this.categoricalValues = categoricalValues;
        }

        /**
         * @return the weight
         */
        public float getWeight() {
            return weight;
        }

        /**
         * @param weight
         *            the weight to set
         */
        public void setWeight(float weight) {
            this.weight = weight;
        }

        /**
         * @return the ideal
         */
        public float getLabel() {
            return label;
        }

        /**
         * @param ideal
         *            the ideal to set
         */
        public void setLabel(float ideal) {
            this.label = ideal;
        }

    }

}
