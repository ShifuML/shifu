package ml.shifu.shifu.core.dtrain.mtl;

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
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.fs.PathFinder;
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
import java.util.concurrent.*;

/**
 * @author haillu
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 3600)
public class MTLWorker extends
        AbstractWorkerComputable<MTLParams, MTLParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTLWorker.class);

    private ModelConfig modelConfig;

    private List<List<ColumnConfig>> mtlColumnConfigLists = new ArrayList<>();

    protected int inputCount;

    private boolean[] isAfterVarSelect;

    /**
     * input record size, inc one by one.
     */
    protected long count;

    protected long sampleCount;

    private volatile MemoryLimitedList<Data> trainingData;

    private volatile MemoryLimitedList<Data> validationData;

    private Splitter splitter;

    private MultiTaskLearning mtl;

    private int trainerId = 0;

    private int workerThreadCount;

    CompletionService<MTLParams> completionService;

    private boolean[] hasCandidates;

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
    private long positiveSelectedTrainCount;
    private long negativeSelectedTrainCount;
    private long positiveTrainCount;
    private long negativeTrainCount;
    private long positiveValidationCount;
    private long negativeValidationCount;
    private int taskNumber;

    // todo:for eval.
    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        // initialize Hadoop based line (long, string) reader
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<MTLParams, MTLParams> context) {
        LOG.debug("worker init:");
        Properties props = context.getProps();
        LOG.info("props: {}", props);

        SourceType sourceType = SourceType
                .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
        loadConfigs(props, sourceType);

        String delimiter = context.getProps().getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER);
        this.splitter = MapReduceUtils.generateShifuOutputSplitter(delimiter);

        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
            LOG.info("Cross validation is enabled by kCrossValidation: {}.", kCrossValidation);
        }

        this.workerThreadCount = modelConfig.getTrain().getWorkerThreadCount();
        this.completionService = new ExecutorCompletionService<>(Executors.newFixedThreadPool(workerThreadCount));

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
            this.trainingData = new MemoryLimitedList<>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * 0.6), new ArrayList<>());
            this.validationData = new MemoryLimitedList<>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * 0.4), new ArrayList<>());
        } else {
            if(validationRate != 0d) {
                this.trainingData = new MemoryLimitedList<>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction * (1 - validationRate)),
                        new ArrayList<>());
                this.validationData = new MemoryLimitedList<>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction * validationRate), new ArrayList<>());
            } else {
                this.trainingData = new MemoryLimitedList<>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                        new ArrayList<>());
            }
        }

        this.isAfterVarSelect = new boolean[taskNumber];
        this.hasCandidates = new boolean[taskNumber];
        for(int i = 0; i < taskNumber; i++) {
            List<ColumnConfig> ccs = this.mtlColumnConfigLists.get(i);
            int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(ccs);
            // numerical + categorical = # of all input
            this.inputCount += inputOutputIndex[0] + inputOutputIndex[1];
            // regression outputNodeCount is 1, binaryClassfication, it is 1, OneVsAll it is 1, Native classification it
            // is
            // 1, with index of 0,1,2,3 denotes different classes
            this.isAfterVarSelect[i] = (inputOutputIndex[3] == 1);
            this.hasCandidates[i] = CommonUtils.hasCandidateColumns(ccs);

        }

        this.isManualValidation = (modelConfig.getValidationDataSetRawPath() != null
                && !"".equals(modelConfig.getValidationDataSetRawPath()));

        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();

        // build multi-task learning model:
        this.validParams = this.modelConfig.getTrain().getParams();
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        List<String> hiddenActiFuncs = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        // todo:check if MTL need regression function
        // double l2reg = NumberUtils.toDouble(this.validParams.get(CommonConstants.WDL_L2_REG).toString(), 0);

        LOG.debug("params of constructor of MTL:inputCount:{},hiddenNodes:{},hiddenActiFuncs:{}" + "taskNumber:{}",
                inputCount, hiddenNodes, hiddenActiFuncs, taskNumber);

        this.mtl = new MultiTaskLearning(inputCount, hiddenNodes, hiddenActiFuncs, taskNumber, 0d);

    }

    private void loadConfigs(Properties props, SourceType sourceType) {
        try {

            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);

            // build mtlColumnConfigLists.
            List<String> tagColumns = this.modelConfig.getMultiTaskTargetColumnNames();
            taskNumber = tagColumns.size();
            LOG.info("tagColumns:{}", tagColumns);

            PathFinder pf = new PathFinder(this.modelConfig);
            LOG.info("mtl folder:{}", pf.getMTLColumnConfigFolder(sourceType));

            for(int i = 0; i < taskNumber; i++) {
                List<ColumnConfig> ccs;
                ccs = CommonUtils.loadColumnConfigList(pf.getMTLColumnConfigPath(sourceType, i), sourceType);
                // for local test:
                // ccs = CommonUtils.loadColumnConfigList(
                // "/C:/Users/haillu/Documents/gitRepo/shifu/target/test-classes/model/MultiTaskNN/mtl/ColumnConfig.json."
                // + i,
                // sourceType);

                mtlColumnConfigLists.add(ccs);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public MTLParams doCompute(WorkerContext<MTLParams, MTLParams> context) {
        if(context.isFirstIteration()) {
            return new MTLParams();
        }

        this.mtl.updateWeights(context.getLastMasterResult());
        MTLParallelGradient parallelGradient = new MTLParallelGradient(this.workerThreadCount, this.mtl,
                this.trainingData, this.validationData, this.completionService);
        MTLParams mtlParams = parallelGradient.doCompute();

        mtlParams.setSerializationType(SerializationType.GRADIENTS);
        this.mtl.setSerializationType(SerializationType.GRADIENTS);
        return mtlParams;
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<MTLParams, MTLParams> context) {
        if(++this.count % 5000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        long hashcode = 0; // hashcode for fixed input split in train and validation
        double[] inputs = new double[this.inputCount];
        double[] ideal = new double[this.taskNumber];
        double significance = 1.0;
        int index = 0, targetIndex = 0;
        int totalSize = 0;
        for(int i = 0; i < taskNumber; i++) {
            totalSize += this.mtlColumnConfigLists.get(i).size();
        }
        // use guava Splitter to iterate only once
        for(String input: this.splitter.split(currentValue.getWritable().toString())) {
            // LOG.info("input in worker: {}", input);
            int accumulativeSum = 0;
            int i = 0;
            for(; i < taskNumber; i++) {
                accumulativeSum += mtlColumnConfigLists.get(i).size();
                if(accumulativeSum > index) {
                    break;
                }
            }

            if(index == totalSize) {
                significance = getWeightValue(input);
                break;
            } else {
                int beginPos = accumulativeSum - mtlColumnConfigLists.get(i).size();
                ColumnConfig config = mtlColumnConfigLists.get(i).get(index - beginPos);
                if(config != null && config.isTarget()) {
                    ideal[targetIndex] = getDoubleValue(input);
                    targetIndex++;
                } else {
                    if(validColumn(config, i)) {
                        hashcode = hashcode * 31 + input.hashCode();
                    }
                }
            }
            index++;
        }

        // todo:logic of sampling.
        // // sample negative only logic here, sample negative out, no need continue
        // if(sampleNegOnly(hashcode, ideal)) {
        // return;
        // }
        // // up sampling logic, just add more weights while bagging sampling rate is still not changed
        // if(modelConfig.isRegression() && isUpSampleEnabled() && Double.compare(ideal, 1d) == 0) {
        // // ideal == 1 means positive tags; sample + 1 to avoid sample count to 0
        // significance = significance * (this.upSampleRng.sample() + 1);
        // }

        // todo: some fileds like 'positiveSelectedTrainCount','negativeSelectedTrainCount' should be an array rather
        // than a number.
        Data data = new Data(inputs, significance, ideal);
        // split into validation and training data set according to validation rate
        boolean isInTraining = this.addDataPairToDataSet(hashcode, data, context.getAttachment());
        // update some positive or negative selected count in metrics
        this.updateMetrics(data, isInTraining);
    }

    private void updateMetrics(Data data, boolean isInTraining) {
        // do bagging sampling only for training data
        if(isInTraining) {
            // for training data, compute real selected training data according to baggingSampleRate
            if(isPositive(data.labels)) {
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

    private boolean splitDataPairToDataSet(long hashcode, Data data) {
        if(Double.compare(this.modelConfig.getValidSetRate(), 0d) != 0) {
            // todo: we just use first label to generate random.
            Random random = updateRandom((int) (data.labels[0] + 0.01d));
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

    private void updateTrainPosNegMetrics(Data data) {

        if(isPositive(data.labels)) {
            this.positiveTrainCount += 1L;
        } else {
            this.negativeTrainCount += 1L;
        }
    }

    private void updateValidationPosNegMetrics(Data data) {

        if(isPositive(data.labels)) {
            this.positiveValidationCount += 1L;
        } else {
            this.negativeValidationCount += 1L;
        }
    }

    private boolean isPositive(double[] value) {
        boolean ret = true;
        for(int i = 0; i < value.length; i++) {
            if(Double.compare(1d, value[i]) != 0) {
                ret = false;
                break;
            }
        }
        return ret;
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

    protected boolean isUpSampleEnabled() {
        // only enabled in regression
        return this.upSampleRng != null && (modelConfig.isRegression()
                || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()));
    }

    /**
     * If column is valid and be selected in model training
     * 
     * @param columnConfig
     *            specific column
     * @param index
     *            index of tasks.
     * @return if it is valid ,return true, others are false.
     */
    private boolean validColumn(ColumnConfig columnConfig, int index) {
        if(isAfterVarSelect[index]) {
            return columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                    && columnConfig.isFinalSelect();
        } else {
            return !columnConfig.isMeta() && !columnConfig.isTarget()
                    && CommonUtils.isGoodCandidate(columnConfig, this.hasCandidates[index]);
        }
    }

    private double getDoubleValue(String input) {
        // check here to avoid bad performance in failed NumberFormatUtils.getDouble(input, 0d)
        double doubleValue = input.length() == 0 ? 0d : NumberFormatUtils.getDouble(input, 0d);
        // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
        return (Double.isNaN(doubleValue) || Double.isNaN(doubleValue)) ? 0d : doubleValue;
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

    @Override
    protected void postLoad(WorkerContext<MTLParams, MTLParams> context) {
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

    public static class Data {
        /**
         * All input values
         */
        private double[] inputs;

        /**
         * The weight of one training recordlike dollar amounts in one txn
         */
        private double weight;

        /**
         * Target value of one record
         */
        private double labels[];

        public Data(double[] inputs, double weight, double[] labels) {
            this.inputs = inputs;
            this.weight = weight;
            this.labels = labels;
        }

        public double[] getInputs() {
            return inputs;
        }

        public void setInputs(double[] inputs) {
            this.inputs = inputs;
        }

        public double getWeight() {
            return weight;
        }

        public void setWeights(double weight) {
            this.weight = weight;
        }

        public double[] getLabels() {
            return labels;
        }

        public void setLabels(double[] labels) {
            this.labels = labels;
        }
    }

}
