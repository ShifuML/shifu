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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.encog.mathutil.BoundMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.guagua.worker.WorkerContext.WorkerCompletionCallBack;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;
import ml.shifu.shifu.core.dtrain.layer.SparseInput;
import ml.shifu.shifu.core.dtrain.loss.LossType;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.MapReduceUtils;

/**
 * {@link WDLWorker} is responsible for loading part of data into memory, do iteration gradients computation and send
 * back to master for master aggregation. After master aggregation is done, received latest weights to do next
 * iteration.
 * 
 * <p>
 * Data loading into memory as memory list includes two parts: numerical double array and sparse input object array
 * which is for categorical variables. To leverage sparse feature of categorical variables, sparse object is leveraged
 * to save memory and matrix computation.
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
     * Worker thread count used as multiple threading to get node status
     */
    private int workerThreadCount;

    /**
     * CompletionService to running gradient update in parallel
     */
    CompletionService<WDLParams> completionService;

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
     * Log(cross entropy) or squared loss definition.
     */
    private LossType lossType;

    private int batchs;

    private boolean isLog = true;

    /**
     * Logic to load data into memory list which includes double array for numerical features and sparse object array
     * for
     * categorical features.
     */
    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<WDLParams, WDLParams> context) {
        if((++this.count) % 5000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        long hashcode = 0; // hashcode for fixed input split in train and validation
        double[] inputs = null;
        SparseInput[] cateInputs = null;
        double ideal = 0d, significance = 1d;
        int index = 0, numIndex = 0, cateIndex = 0;
        // use guava Splitter to iterate only once
        switch(this.modelConfig.getNormalizeType()) {
            case ZSCALE_APPEND_INDEX:
            case ZSCORE_APPEND_INDEX:
            case WOE_APPEND_INDEX:
            case WOE_ZSCALE_APPEND_INDEX:
                inputs = new double[this.numInputs + this.cateInputs];
                cateInputs = new SparseInput[this.numInputs + this.cateInputs];
                if(isLog) {
                    LOG.info("denseinput of data {}, cate input of data {}", inputs.length, cateInputs.length);
                }
                List<String> list = CommonUtils.splitAndReturnList(currentValue.getWritable().toString(),
                        this.splitter);
                for(int i = 0; i < list.size(); i++) {
                    String firstInput = list.get(i);
                    if(i == list.size() - 1) {
                        significance = getWeightValue(firstInput);
                        continue;
                    }

                    ColumnConfig config = this.columnConfigList.get(index++);

                    if(config.isMeta()) {
                        continue; // metadata, skip this one and go to next i
                    } else if(config != null && config.isTarget()) {
                        ideal = getDoubleValue(firstInput);
                    } else {
                        // final select some variables but meta and target are not included
                        if(validColumn(config)) {
                            inputs[numIndex] = getDoubleValue(firstInput);
                            this.inputIndexMap.putIfAbsent(config.getColumnNum(), numIndex++);
                            hashcode = hashcode * 31 + firstInput.hashCode();

                            String secondInput = list.get(i + 1);
                            cateInputs[cateIndex] = new SparseInput(config.getColumnNum(),
                                    (int) getDoubleValue(secondInput));
                            this.inputIndexMap.putIfAbsent(config.getColumnNum(), cateIndex++);
                            hashcode = hashcode * 31 + secondInput.hashCode();
                        }
                        i += 1;
                    }
                }
                break;
            default:
                inputs = new double[this.numInputs];
                cateInputs = new SparseInput[this.cateInputs];

                for(String input: this.splitter.split(currentValue.getWritable().toString())) {
                    // if no wgt column at last pos, no need process here
                    if(index == this.columnConfigList.size()) {
                        significance = getWeightValue(input);
                        break; // the last field is significance, break here
                    } else {
                        ColumnConfig config = this.columnConfigList.get(index);
                        if(config != null && config.isTarget()) {
                            ideal = getDoubleValue(input);
                        } else {
                            // final select some variables but meta and target are not included
                            if(validColumn(config)) {
                                if(config.isNumerical()) {
                                    inputs[numIndex] = getDoubleValue(input);
                                    this.inputIndexMap.putIfAbsent(config.getColumnNum(), numIndex++);
                                } else if(config.isCategorical()) {
                                    cateInputs[cateIndex] = new SparseInput(config.getColumnNum(),
                                            (int) getDoubleValue(input));
                                    this.inputIndexMap.putIfAbsent(config.getColumnNum(), cateIndex++);
                                }
                                hashcode = hashcode * 31 + input.hashCode();
                            }
                        }
                    }
                    index += 1;
                }
                break;
        }

        // output delimiter in norm can be set by user now and if user set a special one later changed, this exception
        // is helped to quick find such issue, here only check numerical array
        validateInputLength(context, inputs, numIndex);

        // sample negative only logic here, sample negative out, no need continue
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
        isLog = false;
    }

    private boolean miniBatchEnabled() {
        return null != this.modelConfig.getTrain().getParams().get(CommonConstants.MINI_BATCH);
    }

    protected boolean isUpSampleEnabled() {
        // only enabled in regression
        return this.upSampleRng != null && (modelConfig.isRegression()
                || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()));
    }

    private boolean sampleNegOnly(long hashcode, double ideal) {
        if(!modelConfig.getTrain().getSampleNegOnly()) { // only works when set sampleNegOnly
            return false;
        }
        boolean ret = false;
        double bagSampleRate = this.modelConfig.getBaggingSampleRate();
        if(this.modelConfig.isFixInitialInput()) {
            // if fixInitialInput, sample hashcode in 1-sampleRate range out if negative records
            int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
            // here BaggingSampleRate means how many data will be used in training and validation, if it is 0.8, we
            // should take 1-0.8 to check endHashCode
            int endHashCode = startHashCode + (int) ((1d - bagSampleRate) * 100);
            if((modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()))
                    && (int) (ideal + 0.01d) == 0 && isInRange(hashcode, startHashCode, endHashCode)) {
                ret = true;
            }
        } else {
            // if not fixed initial input, for regression or onevsall multiple classification, if negative record
            if((modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()))
                    && (int) (ideal + 0.01d) == 0 && negOnlyRnd.nextDouble() > bagSampleRate) {
                ret = true;
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
            return addKFoldDataPairToDataSet(hashcode, data);
        }

        if(this.isManualValidation) { // validation data set is set by users in ModelConfig:dataSet:validationDataPath
            return addManualValidationDataPairToDataSet(data, isValidation);
        } else { // normal case, according to validSetRate, split dataset into training and validation data set
            return splitDataPairToDataSet(hashcode, data);
        }
    }

    private boolean splitDataPairToDataSet(long hashcode, Data data) {
        if(Double.compare(this.modelConfig.getValidSetRate(), 0d) != 0) {
            Random random = updateRandom((int) (data.label + 0.01d));
            if(this.modelConfig.isFixInitialInput()) {
                // for fix initial input, if hashcode%100 is in [start-hashcode, end-hashcode), validation,
                // otherwise training. start hashcode in different job is different to make sure bagging jobs have
                // different data. if end-hashcode is over 100, then check if hashcode is in [start-hashcode, 100]
                // or [0, end-hashcode]
                int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                int endHashCode = startHashCode + (int) (this.modelConfig.getValidSetRate() * 100);
                if(isInRange(hashcode, startHashCode, endHashCode)) {
                    this.validationData.append(data);
                    updateValidationPosNegMetrics(data);
                    return false;
                } else {
                    this.trainingData.append(data);
                    updateTrainPosNegMetrics(data);
                    return true;
                }
            } else {
                // not fixed initial input, if random value >= validRate, training, otherwise validation.
                if(random.nextDouble() >= this.modelConfig.getValidSetRate()) {
                    this.trainingData.append(data);
                    updateTrainPosNegMetrics(data);
                    return true;
                } else {
                    this.validationData.append(data);
                    updateValidationPosNegMetrics(data);
                    return false;
                }
            }
        } else {
            this.trainingData.append(data);
            updateTrainPosNegMetrics(data);
            return true;
        }
    }

    private boolean addManualValidationDataPairToDataSet(Data data, boolean isValidation) {
        if(isValidation) {
            this.validationData.append(data);
            updateValidationPosNegMetrics(data);
            return false;
        } else {
            this.trainingData.append(data);
            updateTrainPosNegMetrics(data);
            return true;
        }
    }

    private boolean addKFoldDataPairToDataSet(long hashcode, Data data) {
        int k = this.modelConfig.getTrain().getNumKFold();
        if(hashcode % k == this.trainerId) {
            this.validationData.append(data);
            updateValidationPosNegMetrics(data);
            return false;
        } else {
            this.trainingData.append(data);
            updateTrainPosNegMetrics(data);
            return true;
        }
    }

    private Random updateRandom(int classValue) {
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
        return random;
    }

    private void updateTrainPosNegMetrics(Data data) {
        if(isPositive(data.label)) {
            this.positiveTrainCount += 1L;
        } else {
            this.negativeTrainCount += 1L;
        }
    }

    private void updateValidationPosNegMetrics(Data data) {
        if(isPositive(data.label)) {
            this.positiveValidationCount += 1L;
        } else {
            this.negativeValidationCount += 1L;
        }
    }

    private boolean isPositive(double value) {
        return Double.compare(1d, value) == 0;
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
    private void validateInputLength(WorkerContext<WDLParams, WDLParams> context, double[] inputs, int numInputIndex) {
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

    @SuppressWarnings("unused")
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

    private double getWeightValue(String input) {
        double significance = 1d;
        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())) {
            // check here to avoid bad performance in failed NumberFormatUtils.getDouble(input, 1d)
            significance = input.length() == 0 ? 1d : NumberFormatUtils.getDouble(input, 1d);
            // if invalid weight, set it to 1d and warning in log
            if(significance < 0d) {
                LOG.warn("Record {} with weight {} is less than 0 and invalid, set it to 1.", count, significance);
                significance = 1d;
            }
        }
        return significance;
    }

    private double getDoubleValue(String input) {
        // check here to avoid bad performance in failed NumberFormatUtils.getDouble(input, 0d)
        double doubleValue = input.length() == 0 ? 0d : NumberFormatUtils.getDouble(input, 0d);
        // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
        return (Double.isNaN(doubleValue) || Double.isNaN(doubleValue)) ? 0d : doubleValue;
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

        this.workerThreadCount = modelConfig.getTrain().getWorkerThreadCount();
        final ExecutorService fixedThreadPool = Executors.newFixedThreadPool(workerThreadCount);
        this.completionService = new ExecutorCompletionService<>(fixedThreadPool);

        // register call back for shut down thread pool.
        context.addCompletionCallBack(new WorkerCompletionCallBack<WDLParams, WDLParams>() {
            @Override
            public void callback(WorkerContext<WDLParams, WDLParams> context) {
                fixedThreadPool.shutdown();
            }
        });

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
        this.cateInputs = inputOutputIndex[1];
        this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];

        // regression outputNodeCount is 1, binaryClassfication, it is 1, OneVsAll it is 1, Native classification it is
        // 1, with index of 0,1,2,3 denotes different classes
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        this.isManualValidation = (modelConfig.getValidationDataSetRawPath() != null
                && !"".equals(modelConfig.getValidationDataSetRawPath()));

        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();

        this.validParams = this.modelConfig.getTrain().getParams();

        Object miniBatchO = validParams.get(CommonConstants.MINI_BATCH);
        if(miniBatchO != null) {
            int miniBatchs;
            try {
                miniBatchs = Integer.parseInt(miniBatchO.toString());
            } catch (Exception e) {
                miniBatchs = 1;
            }
            if(miniBatchs < 0) {
                this.batchs = 1;
            } else if(miniBatchs > 1000) {
                this.batchs = 1000;
            } else {
                this.batchs = miniBatchs;
            }
            LOG.info("'miniBatchs' in worker is : {}, batchs is {} ", miniBatchs, batchs);
        }

        Object lossObj = validParams.get("Loss");
        this.lossType = LossType.of(lossObj != null ? lossObj.toString() : CommonConstants.SQUARED_LOSS);
        LOG.info("Loss type is {}.", this.lossType);

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
        double l2reg = NumberUtils.toDouble(this.validParams.get(CommonConstants.L2_REG).toString(), 0d);
        Object wideEnableObj = this.validParams.get(CommonConstants.WIDE_ENABLE);
        boolean wideEnable = CommonUtils.getBooleanValue(this.validParams.get(CommonConstants.WIDE_ENABLE), true);
        boolean deepEnable = CommonUtils.getBooleanValue(this.validParams.get(CommonConstants.DEEP_ENABLE), true);
        boolean embedEnable = CommonUtils.getBooleanValue(this.validParams.get(CommonConstants.EMBED_ENABLE), true);
        boolean wideDenseEnable = CommonUtils.getBooleanValue(this.validParams.get(CommonConstants.WIDE_DENSE_ENABLE),
                true);
        NormType normType = this.modelConfig.getNormalizeType();

        int deepNumInputs = this.numInputs;
        if(NormType.ZSCALE_APPEND_INDEX.equals(normType) || NormType.ZSCORE_APPEND_INDEX.equals(normType)
                || NormType.WOE_APPEND_INDEX.equals(normType) || NormType.WOE_ZSCALE_APPEND_INDEX.equals(normType)) {
            deepNumInputs = this.inputCount;
            numericalIds.addAll(wideColumnIds);
            embedColumnIds = new ArrayList<Integer>();
            embedOutputList = new ArrayList<Integer>();
            for(Integer id: numericalIds) {
                embedColumnIds.add(id);
                embedOutputList.add(embedOutputs == null ? CommonConstants.DEFAULT_EMBEDING_OUTPUT : embedOutputs);
            }
            Collections.sort(embedColumnIds);
            LOG.info("deepNumInputs {}; numericalIds {}; embedColumnIds {}.", deepNumInputs, numericalIds.size(),
                    embedColumnIds.size());
        }
        this.wnd = new WideAndDeep(wideEnable, deepEnable, embedEnable, wideDenseEnable, idBinCateSizeMap,
                deepNumInputs, numericalIds, embedColumnIds, embedOutputList, wideColumnIds, hiddenNodes, actFunc,
                l2reg);
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
        WDLParallelGradient parallelGradient = new WDLParallelGradient(this.wnd, this.workerThreadCount,
                this.inputIndexMap, this.trainingData, this.validationData, this.completionService, this.lossType);
        WDLParams wdlParams = null;
        if(miniBatchEnabled()) {
            int iteration = context.getCurrentIteration();
            int miniBatchSize = Integer
                    .parseInt(this.modelConfig.getTrain().getParams().get(CommonConstants.MINI_BATCH).toString());
            wdlParams = parallelGradient.doCompute(iteration, miniBatchSize);
        } else {
            wdlParams = parallelGradient.doCompute();
        }
        wdlParams.setSerializationType(SerializationType.GRADIENTS);
        this.wnd.setSerializationType(SerializationType.GRADIENTS);
        return wdlParams;
    }

    public double sigmoid(double logit) {
        // return (double) (1 / (1 + Math.min(1.0E19, Math.exp(-logit))));
        return 1.0d / (1.0d + BoundMath.exp(-1 * logit));
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
     * {@link Data} denotes training record with a double array of dense (numerical) inputs and a list of sparse inputs
     * of categorical input features.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    public static class Data {

        /**
         * Numerical values
         */
        private double[] numericalValues;

        /**
         * Categorical values in sparse object
         */
        private SparseInput[] categoricalValues;

        /**
         * The weight of one training record like dollar amount in one txn
         */
        private double weight;

        /**
         * Target value of one record
         */
        private double label;

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
        public Data(double[] numericalValues, SparseInput[] categoricalValues, double weight, double ideal) {
            this.numericalValues = numericalValues;
            this.categoricalValues = categoricalValues;
            this.weight = weight;
            this.label = ideal;
        }

        /**
         * @return the numericalValues
         */
        public double[] getNumericalValues() {
            return numericalValues;
        }

        /**
         * @param numericalValues
         *            the numericalValues to set
         */
        public void setNumericalValues(double[] numericalValues) {
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
        public double getWeight() {
            return weight;
        }

        /**
         * @param weight
         *            the weight to set
         */
        public void setWeight(double weight) {
            this.weight = weight;
        }

        /**
         * @return the ideal
         */
        public double getLabel() {
            return label;
        }

        /**
         * @param ideal
         *            the ideal to set
         */
        public void setLabel(double ideal) {
            this.label = ideal;
        }

    }

}
