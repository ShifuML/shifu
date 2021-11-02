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
package ml.shifu.shifu.core.dtrain.mtl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLData;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataSet;
import ml.shifu.shifu.core.dtrain.dataset.MemoryDiskFloatMLDataSet;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.MapReduceUtils;

/**
 * {@link MTLWorker} is worker role in master-workers iterative computing for multi-task model distributed training.
 * 
 * <p>
 * Normalized data is stored into one line for multiple tasks one by one: if 500 columns and 3 tasks, the data of each
 * line is organized by 500 normalized values + 2nd 500 normalized values + 3rd 500 normalized values + weight column.
 * 
 * <p>
 * In each epoch, worker receives global model weights from master and updates local model instance. Based on latest
 * model weights, by iterating local training data to accumulate gradients in parallel and send back to master. This is
 * typical one epoch for one worker.
 * 
 * <p>
 * TODO, mini-batch support?
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MTLWorker extends
        AbstractWorkerComputable<MTLParams, MTLParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTLWorker.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration list loaded from multiple json column configuration files.
     */
    protected List<List<ColumnConfig>> mtlColumnConfigLists;

    /**
     * Number of input count used in model training from all multiple ColumnConfig list.
     */
    protected int inputCount;

    /**
     * Means if do variable selection, if done, many variables will be set to finalSelect = true; if not, no variables
     * are selected and should be set to all good candidate variables.
     */
    private boolean isAfterVarSelect = true;

    /**
     * # of input records in such worker.
     */
    protected long count;

    /**
     * If k-fold cross validation.
     */
    private boolean isKFoldCV;

    /**
     * Training data set with only in memory, in the future, MemoryDiskList can be leveraged.
     */
    private volatile FloatMLDataSet trainingData;

    /**
     * Validation data set with only in memory.
     */
    private volatile FloatMLDataSet validationData;

    /**
     * A splitter to split data with specified delimiter.
     */
    private Splitter splitter;

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
    private CompletionService<MTLParams> completionService;

    /**
     * If has candidate in column list.
     */
    private boolean hasCandidates;

    /**
     * Indicates if validation are set by users for validationDataPath, not random picking
     */
    protected boolean isManualValidation = false;

    /**
     * A validation random map for different classes, stratified sampling, is useful for class level sampling.
     */
    private Map<Integer, Random> validationRandomMap = new HashMap<Integer, Random>();

    /**
     * Whether to enable poisson bagging with replacement.
     */
    protected boolean poissonSampler;

    /**
     * PoissonDistribution which is used for poisson sampling for bagging with replacement.
     */
    protected PoissonDistribution rng = null;

    /**
     * Parameters defined in ModelConfig.json#train part.
     */
    private Map<String, Object> validParams;

    /**
     * MutliTaskModel instance for model graph definition.
     */
    private MultiTaskModel mtm;

    /**
     * Tag column names for multiple tasks.
     */
    private List<String> multiTagColumns;

    /**
     * If has multiple weight columns in MTL
     */
    private boolean isMultiWeights;

    /**
     * Load data as input data array into memory training data set and validation data set. Multiple tasks are sorted
     * one by one in the same line to avoid join operations.
     */
    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<MTLParams, MTLParams> context) {
        if((++this.count) % 5000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        long hashcode = 0; // hashcode for fixed input split in train and validation
        float[] inputs = new float[this.inputCount];
        float[] outputs = new float[this.multiTagColumns.size()];

        float significance = 1f;
        float[] significances = new float[this.multiTagColumns.size()];

        int index = 0, inputIndex = 0, outputIndex = 0, mtlWgtIndex = 0;

        // all columnConfigList have the same size, so just get the first one
        int columns = this.mtlColumnConfigLists.get(0).size();

        // use guava Splitter to iterate only once
        for(String input: this.splitter.split(currentValue.getWritable().toString())) {
            if(this.isMultiWeights) {
                if(index >= this.multiTagColumns.size() * columns) {
                    significances[mtlWgtIndex++] = getWeightValue(input);
                    if(mtlWgtIndex < significances.length) {
                        continue;
                    } else {
                        break; // last fields are all weights
                    }
                }
            } else {
                if(index == this.multiTagColumns.size() * columns) {
                    significance = getWeightValue(input);
                    break; // only the last field is significance, break here
                }
            }

            // multiple norm outputs are appended one by one in the same line
            int currCCListIndex = index / columns, currCCIndex = index % columns;
            ColumnConfig config = this.mtlColumnConfigLists.get(currCCListIndex).get(currCCIndex);
            if(config != null && config.isTarget()) {
                outputs[outputIndex++] = CommonUtils.getFloatValue(input);
            } else {
                // final select some variables but meta and target are not included
                if(validColumn(config)) {
                    inputs[inputIndex++] = CommonUtils.getFloatValue(input);
                    hashcode = hashcode * 31 + input.hashCode();
                }
            }
            index += 1;
        }

        // output delimiter in norm can be set by user now and if user set a special one later changed, this exception
        // is helped to quick find such issue.
        validateInputLength(context, inputs, inputIndex);

        FloatMLDataPair data = new BasicFloatMLDataPair(new BasicFloatMLData(inputs), new BasicFloatMLData(outputs));
        data.setSignificance(significance);
        if(this.isMultiWeights) {
            data.setSignificances(significances);
        }

        // split into validation and training data set according to validation rate
        this.addDataPairToDataSet(hashcode, data, context.getAttachment());
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
     * @return if in training, training is true, false for validation.
     */
    protected boolean addDataPairToDataSet(long hashcode, FloatMLDataPair data, Object attachment) {
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

    private boolean splitDataPairToDataSet(long hashcode, FloatMLDataPair data) {
        if(Double.compare(this.modelConfig.getValidSetRate(), 0d) != 0) {
            Random random = updateRandom((int) (data.getIdeal().getData()[0]));
            if(this.modelConfig.isFixInitialInput()) {
                // for fix initial input, if hashcode%100 is in [start-hashcode, end-hashcode), validation,
                // otherwise training. start hashcode in different job is different to make sure bagging jobs have
                // different data. if end-hashcode is over 100, then check if hashcode is in [start-hashcode, 100]
                // or [0, end-hashcode]
                int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                int endHashCode = startHashCode + (int) (this.modelConfig.getValidSetRate() * 100);
                if(isInRange(hashcode, startHashCode, endHashCode)) {
                    this.validationData.add(data);
                    return false;
                } else {
                    this.trainingData.add(data);
                    return true;
                }
            } else {
                // not fixed initial input, if random value >= validRate, training, otherwise validation.
                if(random.nextDouble() >= this.modelConfig.getValidSetRate()) {
                    this.trainingData.add(data);
                    return true;
                } else {
                    this.validationData.add(data);
                    return false;
                }
            }
        } else {
            this.trainingData.add(data);
            return true;
        }
    }

    private Random updateRandom(int classValue) {
        Random random = validationRandomMap.get(0);
        if(random == null) {
            random = new Random();
            this.validationRandomMap.put(0, random);
        }
        return random;
    }

    /**
     * Add data pair into training or validation data set. Validation data from customized validation data path.
     * 
     * @param data
     *            data pair record
     * @param isValidation
     *            if is validation data set
     * @return if in training data set
     */
    private boolean addManualValidationDataPairToDataSet(FloatMLDataPair data, boolean isValidation) {
        if(isValidation) {
            this.validationData.add(data);
        } else {
            this.trainingData.add(data);
        }
        return !isValidation;
    }

    /**
     * Add data pair into training or validation data set.
     * 
     * @param hashcode
     *            the hashcode of the line
     * @param data
     *            data pair record
     * @return if in training data set
     */
    private boolean addKFoldDataPairToDataSet(long hashcode, FloatMLDataPair data) {
        int k = this.modelConfig.getTrain().getNumKFold();
        if(hashcode % k == this.trainerId) {
            this.validationData.add(data);
            return false;
        } else {
            this.trainingData.add(data);
            return true;
        }
    }

    private boolean isInRange(long hashcode, int startHashCode, int endHashCode) {
        // check if in [start, end] or if in [start, 100) and [0, end-100)
        int hashCodeIn100 = (int) hashcode % 100;
        if(endHashCode <= 100) {
            // in range [start, end)
            return hashCodeIn100 >= startHashCode && hashCodeIn100 < endHashCode;
        } else {
            // in range [start, 100) or [0, end-100)
            return hashCodeIn100 >= startHashCode || hashCodeIn100 < (endHashCode % 100);
        }
    }

    /**
     * If no enough columns for model training, most of the cases root cause is from inconsistent delimiter.
     */
    private void validateInputLength(WorkerContext<MTLParams, MTLParams> context, float[] inputs, int numInputIndex) {
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

    private float getWeightValue(String input) {
        float significance = 1f;
        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())) {
            // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 1f)
            significance = input.length() == 0 ? 1f : NumberFormatUtils.getFloat(input, 1f);
            // if invalid weight, set it to 1d and warning in log
            if(significance < 0f) {
                LOG.warn("Record {} with weight {} is less than 0 and invalid, set it to 1.", count, significance);
                significance = 1f;
            }
        }
        return significance;
    }

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        // initialize Hadoop based line (long, string) reader
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @SuppressWarnings({ "unchecked" })
    @Override
    public void init(WorkerContext<MTLParams, MTLParams> context) {
        Properties props = context.getProps();
        loadConfigs(props);

        this.isMultiWeights = this.modelConfig.isMultiTask() && this.modelConfig.isMultiWeightsInMTL();

        this.hasCandidates = CommonUtils.hasCandidateColumns(this.mtlColumnConfigLists.get(0));

        // create Splitter
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

        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));

        initDataSet(context);

        this.inputCount = 0;
        for(List<ColumnConfig> columnConfigList: mtlColumnConfigLists) {
            int[] inputsAndOutpus = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
            this.inputCount += inputsAndOutpus[0] + inputsAndOutpus[1];
        }

        this.isManualValidation = (modelConfig.getValidationDataSetRawPath() != null
                && !"".equals(modelConfig.getValidationDataSetRawPath()));

        this.validParams = this.modelConfig.getTrain().getParams();

        // Build multiple task model architecture
        List<String> actiFuncs = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodes = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);
        double l2reg = NumberUtils.toDouble(this.validParams.get(CommonConstants.L2_REG).toString(), 0d);
        List<Integer> finalOutputs = new ArrayList<>();
        int tasks = this.modelConfig.getMultiTaskTargetColumnNames().size();
        for(int i = 0; i < tasks; i++) {
            finalOutputs.add(1);
        }
        this.mtm = new MultiTaskModel(this.inputCount, hiddenNodes, actiFuncs, finalOutputs, l2reg);

        // Init multiple task model optimizer
        double learningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());
        Object pObject = this.validParams.get(CommonConstants.PROPAGATION);
        String propagation = (pObject == null) ? DTrainUtils.RESILIENTPROPAGATION : pObject.toString();
        // l2 hard code to NONE here because already set in MultiTaskModel backward
        this.mtm.initOptimizer(learningRate, propagation, 0, RegulationLevel.NONE);
    }

    /**
     * Initialize training and validation data set according to memory fraction rate.
     */
    private void initDataSet(WorkerContext<MTLParams, MTLParams> context) {
        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("NNWorker is loading data into memory.");
        long memoryStoreSize = (long) (Runtime.getRuntime().maxMemory() * memoryFraction);
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);
        double valdSetRate = this.modelConfig.getValidSetRate();
        try {
            if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
                // fixed 0.6 and 0.4 of max memory for trainingData and validationData
                this.trainingData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * 0.6),
                        DTrainUtils.getTrainingFile().toString(), this.inputCount, this.multiTagColumns.size());
                this.validationData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * 0.4),
                        DTrainUtils.getTestingFile().toString(), this.inputCount, this.multiTagColumns.size());
            } else {
                this.trainingData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * (1 - valdSetRate)),
                        DTrainUtils.getTrainingFile().toString(), this.inputCount, this.multiTagColumns.size());
                this.validationData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * valdSetRate),
                        DTrainUtils.getTestingFile().toString(), this.inputCount, this.multiTagColumns.size());
            }
            // cannot find a good place to close these two data set, using Shutdown hook
            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    ((MemoryDiskFloatMLDataSet) (MTLWorker.this.trainingData)).close();
                    ((MemoryDiskFloatMLDataSet) (MTLWorker.this.validationData)).close();
                }
            }));
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        }
    }

    /**
     * Load ModelConfig.json and all ColumnConfig.json files.
     */
    private void loadConfigs(Properties props) {
        this.mtlColumnConfigLists = new ArrayList<>();

        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.multiTagColumns = this.modelConfig.getMultiTaskTargetColumnNames();
            assert this.multiTagColumns != null && this.multiTagColumns.size() > 0;

            PathFinder pathFinder = new PathFinder(this.modelConfig);
            int ccSize = -1;
            for(int i = 0; i < this.multiTagColumns.size(); i++) {
                String ccPath = pathFinder.getMTLColumnConfigPath(sourceType, i);
                List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(ccPath, sourceType);
                if(ccSize == -1) {
                    ccSize = columnConfigList.size();
                } else {
                    if(ccSize != columnConfigList.size()) {
                        throw new IllegalArgumentException(
                                "Multiple tasks have different columns in ColumnConfig.json files, please check.");
                    }
                }
                this.mtlColumnConfigLists.add(columnConfigList);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Synchronized from latest model of master then do parallel forward-backward computing, return gradients to master.
     */
    @Override
    public MTLParams doCompute(WorkerContext<MTLParams, MTLParams> context) {
        if(context.isFirstIteration()) {
            // return empty which has been ignored in master first iteration, worker needs sync with master at first.
            return new MTLParams();
        }

        // update master global model into worker MTL graph
        this.mtm.updateWeights(context.getLastMasterResult());

        MTLParallelGradient parallelGradient = new MTLParallelGradient(this.mtm, this.workerThreadCount,
                this.trainingData, this.validationData, this.completionService);
        MTLParams mtmParams = parallelGradient.doCompute();
        mtmParams.setSerializationType(SerializationType.GRADIENTS);
        this.mtm.setSerializationType(SerializationType.GRADIENTS);
        return mtmParams;
    }

    /**
     * Data statistics after loading into memory.
     */
    @Override
    protected void postLoad(WorkerContext<MTLParams, MTLParams> context) {
        ((MemoryDiskFloatMLDataSet) this.trainingData).endLoad();
        ((MemoryDiskFloatMLDataSet) this.validationData).endLoad();

        LOG.info("    - # Training Records in memory: {}.",
                ((MemoryDiskFloatMLDataSet) this.trainingData).getMemoryCount());
        LOG.info("    - # Training Records in disk: {}.",
                ((MemoryDiskFloatMLDataSet) this.trainingData).getDiskCount());

        LOG.info("    - # Records of the Total Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        if(this.isKFoldCV) {
            LOG.info("        - Validation Rate(kFold): {}.", 1d / this.modelConfig.getTrain().getNumKFold());
        } else {
            LOG.info("        - Validation Rate: {}.", this.modelConfig.getValidSetRate());
        }
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.getRecordCount());

        if(validationData != null) {
            LOG.info("        - # Records of the Validation Set: {}.", this.validationData.getRecordCount());
        }
    }

}