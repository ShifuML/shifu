/**
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
package ml.shifu.shifu.core.dtrain.nn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.guagua.worker.WorkerContext.WorkerCompletionCallBack;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLData;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataSet;
import ml.shifu.shifu.core.dtrain.dataset.BufferedFloatMLDataSet;
import ml.shifu.shifu.core.dtrain.dataset.FloatFlatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataSet;
import ml.shifu.shifu.core.dtrain.dataset.MemoryDiskFloatMLDataSet;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.error.LinearErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link AbstractNNWorker} is refactored as a common class for different NN input format.
 */
public abstract class AbstractNNWorker<VALUE extends Writable> extends
        AbstractWorkerComputable<NNParams, NNParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<VALUE>> {

    protected static final Logger LOG = LoggerFactory.getLogger(AbstractNNWorker.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    protected static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR)
            .trimResults();

    /**
     * Training data set
     */
    protected FloatMLDataSet trainingData = null;

    /**
     * Validation data set
     */
    protected FloatMLDataSet validationData = null;

    /**
     * NN algorithm runner instance.
     */
    protected ParallelGradient gradient;

    /**
     * Model Config read from HDFS
     */
    protected ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    protected List<ColumnConfig> columnConfigList;

    /**
     * Basic input node count for NN model
     */
    protected int inputNodeCount;

    /**
     * Basic output node count for NN model
     */
    protected int outputNodeCount;

    /**
     * {@link #candidateCount} is used to check if no variable is selected. If {@link #inputNodeCount} equals
     * {@link #candidateCount}, which means no column is selected or all columns are selected.
     */
    protected int candidateCount;

    /**
     * Trainer id used to tag bagging training job, starting from 0, 1, 2 ...
     */
    protected int trainerId = 0;

    /**
     * input record size, inc one by one.
     */
    protected long count;

    /**
     * Whether the training is dry training.
     */
    protected boolean isDry;

    /**
     * In each iteration, how many epochs will be run.
     */
    protected int epochsPerIteration = 1;

    /**
     * Whether to alternative training and testing elements.
     */
    protected boolean isCrossOver = false;

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
     * A instance from context properties which is from job configuration.
     */
    protected Properties props;

    /**
     * Indicates if there are cross validation data sets.
     */
    protected boolean isSpecificValidation = false;

    /**
     * Valid params specially for grid search
     */
    private Map<String, Object> validParams;

    /**
     * If stratified sampling or random sampling
     */
    protected boolean isStratifiedSampling = false;

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
     * PoissonDistribution which is used for poission sampling for bagging with replacement.
     */

    protected Map<Integer, PoissonDistribution> baggingRngMap = new HashMap<Integer, PoissonDistribution>();

    /**
     * Construct a bagging random map for different classes. For stratified sampling, this is useful for each class
     * sampling.
     */
    protected Map<Integer, Random> baggingRandomMap = new HashMap<Integer, Random>();

    /**
     * Construct a validation random map for different classes. For stratified sampling, this is useful for each class
     * sampling.
     */
    protected Map<Integer, Random> validationRandomMap = new HashMap<Integer, Random>();

    /**
     * Random object to sample negative records
     */
    protected Random sampelNegOnlyRandom = new Random(System.currentTimeMillis() + 1000L);

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;

    /**
     * If enabled by extreme learning machine: https://en.wikipedia.org/wiki/Extreme_learning_machine
     */
    private boolean isELM;

    /**
     * Cache all features with feature index for searching
     */
    protected List<Integer> allFeatures;

    /**
     * Cache subset features with feature index for searching
     */
    protected List<Integer> subFeatures;

    /**
     * Set for sub features to quick check if column is in sub feature list
     */
    protected Set<Integer> subFeatureSet;

    protected int featureInputsCnt;

    /**
     * Dropout rate which is in [0, 1], default it is 0
     */
    private double dropoutRate = 0d;

    /**
     * Loss type: log, squared ...
     */
    private String lossStr;

    /**
     * Weight initializer, can be 'default', 'gaussian' or 'xavier'.
     */
    private String wgtInit;

    /**
     * If miniBatchRate set to 0.1d, {@link #batchs} is 10. It will run 10x iterations for one epochs.
     */
    private int batchs = 1;

    protected boolean isUpSampleEnabled() {
        // only enabled in regression
        return this.upSampleRng != null
                && (modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll()));
    }

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
    private void loadConfigFiles(final Properties props) {
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.isCrossOver = this.modelConfig.getTrain().getIsCrossOver().booleanValue();
            LOG.info("Parameter isCrossOver:{}", this.isCrossOver);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create memory data set object
     */
    @SuppressWarnings("unused")
    private void initMemoryDataSet() {
        this.trainingData = new BasicFloatMLDataSet();
        this.validationData = new BasicFloatMLDataSet();
    }

    /**
     * For disk data set , initialize it with parameters and other work about creating files.
     * 
     * @throws IOException
     *             if any exception on local fs operations.
     * @throws RuntimeException
     *             if error on deleting testing or training file.
     */
    private void initDiskDataSet() throws IOException {
        Path trainingFile = DTrainUtils.getTrainingFile();
        Path testingFile = DTrainUtils.getTestingFile();

        LOG.debug("Use disk to store training data and testing data. Training data file:{}; Testing data file:{} ",
                trainingFile.toString(), testingFile.toString());

        this.trainingData = new BufferedFloatMLDataSet(new File(trainingFile.toString()));
        ((BufferedFloatMLDataSet) this.trainingData).beginLoad(this.featureInputsCnt, getOutputNodeCount());

        this.validationData = new BufferedFloatMLDataSet(new File(testingFile.toString()));
        ((BufferedFloatMLDataSet) this.validationData).beginLoad(this.featureInputsCnt, getOutputNodeCount());
    }

    @Override
    public void init(WorkerContext<NNParams, NNParams> context) {
        // load props firstly
        this.props = context.getProps();

        loadConfigFiles(context.getProps());

        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain()
                .getGridConfigFileContent());
        this.validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            this.validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }

        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
            LOG.info("Cross validation is enabled by kCrossValidation: {}.", kCrossValidation);
        }

        this.poissonSampler = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(NNConstants.NN_POISON_SAMPLER));
        this.rng = new PoissonDistribution(1.0d);
        Double upSampleWeight = modelConfig.getTrain().getUpSampleWeight();
        if(Double.compare(upSampleWeight, 1d) != 0
                && (modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll()))) {
            // set mean to upSampleWeight -1 and get sample + 1to make sure no zero sample value
            LOG.info("Enable up sampling with weight {}.", upSampleWeight);
            this.upSampleRng = new PoissonDistribution(upSampleWeight - 1);
        }
        Integer epochsPerIterationInteger = this.modelConfig.getTrain().getEpochsPerIteration();
        this.epochsPerIteration = epochsPerIterationInteger == null ? 1 : epochsPerIterationInteger.intValue();
        LOG.info("epochsPerIteration in worker is :{}", epochsPerIteration);

        Object elmObject = validParams.get(DTrainUtils.IS_ELM);
        isELM = elmObject == null ? false : "true".equalsIgnoreCase(elmObject.toString());
        LOG.info("Check isELM: {}", isELM);

        Object dropoutRateObj = validParams.get(CommonConstants.DROPOUT_RATE);
        if(dropoutRateObj != null) {
            this.dropoutRate = Double.valueOf(dropoutRateObj.toString());
        }
        LOG.info("'dropoutRate' in worker is :{}", this.dropoutRate);

        Object miniBatchO = validParams.get("MiniBatchs");
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

        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        // if is one vs all classification, outputNodeCount is set to 1
        this.outputNodeCount = modelConfig.isRegression() ? inputOutputIndex[1]
                : (modelConfig.getTrain().isOneVsAll() ? inputOutputIndex[1] : modelConfig.getTags().size());
        this.candidateCount = inputOutputIndex[2];
        boolean isAfterVarSelect = inputOutputIndex[0] != 0;
        LOG.info("Input count {}, output count {}, candidate count {}", inputNodeCount, outputNodeCount, candidateCount);
        // cache all feature list for sampling features
        this.allFeatures = CommonUtils.getAllFeatureList(columnConfigList, isAfterVarSelect);
        String subsetStr = context.getProps().getProperty(CommonConstants.SHIFU_NN_FEATURE_SUBSET);
        if(StringUtils.isBlank(subsetStr)) {
            this.subFeatures = this.allFeatures;
        } else {
            String[] splits = subsetStr.split(",");
            this.subFeatures = new ArrayList<Integer>(splits.length);
            for(String split: splits) {
                int featureIndex = Integer.parseInt(split);
                this.subFeatures.add(featureIndex);
            }
        }
        this.subFeatureSet = new HashSet<Integer>(this.subFeatures);
        LOG.info("subFeatures size is {}", subFeatures.size());
        this.featureInputsCnt = DTrainUtils.getFeatureInputsCnt(this.modelConfig, this.columnConfigList,
                this.subFeatureSet);

        this.wgtInit = "default";
        Object wgtInitObj = validParams.get("WeightInitializer");
        if(wgtInitObj != null) {
            this.wgtInit = wgtInitObj.toString();
        }

        Object lossObj = validParams.get("Loss");
        this.lossStr = lossObj != null ? lossObj.toString() : "squared";
        LOG.info("Loss str is {}", this.lossStr);

        this.isDry = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(CommonConstants.SHIFU_DRY_DTRAIN));
        this.isSpecificValidation = (modelConfig.getValidationDataSetRawPath() != null && !"".equals(modelConfig
                .getValidationDataSetRawPath()));
        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();
        if(isOnDisk()) {
            LOG.info("NNWorker is loading data into disk.");
            try {
                initDiskDataSet();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            // cannot find a good place to close these two data set, using Shutdown hook
            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    ((BufferedFloatMLDataSet) (AbstractNNWorker.this.trainingData)).close();
                    ((BufferedFloatMLDataSet) (AbstractNNWorker.this.validationData)).close();
                }
            }));
        } else {
            LOG.info("NNWorker is loading data into memory.");
            double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
            long memoryStoreSize = (long) (Runtime.getRuntime().maxMemory() * memoryFraction);
            LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);
            double crossValidationRate = this.modelConfig.getValidSetRate();
            try {
                if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
                    // fixed 0.6 and 0.4 of max memory for trainingData and validationData
                    this.trainingData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * 0.6), DTrainUtils
                            .getTrainingFile().toString(), this.featureInputsCnt, this.outputNodeCount);
                    this.validationData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * 0.4), DTrainUtils
                            .getTestingFile().toString(), this.featureInputsCnt, this.outputNodeCount);
                } else {
                    this.trainingData = new MemoryDiskFloatMLDataSet(
                            (long) (memoryStoreSize * (1 - crossValidationRate)), DTrainUtils.getTrainingFile()
                                    .toString(), this.featureInputsCnt, this.outputNodeCount);
                    this.validationData = new MemoryDiskFloatMLDataSet((long) (memoryStoreSize * crossValidationRate),
                            DTrainUtils.getTestingFile().toString(), this.featureInputsCnt, this.outputNodeCount);
                }
                // cannot find a good place to close these two data set, using Shutdown hook
                Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                    @Override
                    public void run() {
                        ((MemoryDiskFloatMLDataSet) (AbstractNNWorker.this.trainingData)).close();
                        ((MemoryDiskFloatMLDataSet) (AbstractNNWorker.this.validationData)).close();
                    }
                }));
            } catch (IOException e) {
                throw new GuaguaRuntimeException(e);
            }
        }
    }

    private boolean isOnDisk() {
        return this.modelConfig.getTrain().getTrainOnDisk() != null
                && this.modelConfig.getTrain().getTrainOnDisk().booleanValue();
    }

    @Override
    public NNParams doCompute(WorkerContext<NNParams, NNParams> context) {
        // For dry option, return empty result.
        // For first iteration, we don't do anything, just wait for master to update weights in next iteration. This
        // make sure all workers in the 1st iteration to get the same weights.
        if(this.isDry || context.isFirstIteration()) {
            return buildEmptyNNParams(context);
        }

        if(context.getLastMasterResult() == null) {
            // This may not happen since master will set initialization weights firstly.
            LOG.warn("Master result of last iteration is null.");
            return null;
        }
        LOG.debug("Set current model with params {}", context.getLastMasterResult());

        // initialize gradients if null
        double[] weights = context.getLastMasterResult().getWeights();
        if(gradient == null) {
            initGradient(this.trainingData, this.validationData, weights, this.isCrossOver);
            // register call back for shut down thread pool.
            context.addCompletionCallBack(new WorkerCompletionCallBack<NNParams, NNParams>() {
                @Override
                public void callback(WorkerContext<NNParams, NNParams> context) {
                    AbstractNNWorker.this.gradient.shutdown();
                }
            });
        } else {
            if(this.isCrossOver) {
                // each iteration reset seed
                this.gradient.setSeed(System.currentTimeMillis());
            }
        }

        this.gradient.getNetwork().setWeights(weights);

        // using the weights from master to train model in current iteration
        double[] gradients = null;
        for(int i = 0; i < epochsPerIteration; i++) {
            gradients = this.gradient.computeGradients(context.getCurrentIteration());
            if(this.epochsPerIteration > 1) {
                this.gradient.resetNetworkWeights();
            }
        }
        // get train errors and test errors
        double trainError = this.gradient.getTrainError();

        long start = System.currentTimeMillis();
        double testError = this.validationData.getRecordCount() > 0 ? (this.gradient.calculateError()) : this.gradient
                .getTrainError();
        LOG.info("Computing test error time: {}ms", (System.currentTimeMillis() - start));

        // if the validation set is 0%, then the validation error should be "N/A"
        LOG.info("NNWorker compute iteration {} (train error {} validation error {})",
                new Object[] { context.getCurrentIteration(), trainError,
                        (this.validationData.getRecordCount() > 0 ? testError : "N/A") });

        NNParams params = new NNParams();
        params.setTestError(testError);
        params.setTrainError(trainError);
        params.setGradients(gradients);
        // prevent null point;
        params.setWeights(new double[0]);
        params.setTrainSize(this.trainingData.getRecordCount());
        params.setCount(count);
        return params;
    }

    @SuppressWarnings("unchecked")
    private void initGradient(FloatMLDataSet training, FloatMLDataSet testing, double[] weights, boolean isCrossOver) {
        int numLayers = (Integer) this.validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) this.validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) this.validParams.get(CommonConstants.NUM_HIDDEN_NODES);

        BasicNetwork network = DTrainUtils.generateNetwork(this.featureInputsCnt, this.outputNodeCount, numLayers,
                actFunc, hiddenNodeList, false, this.dropoutRate, this.wgtInit);
        // use the weights from master
        network.getFlat().setWeights(weights);

        FlatNetwork flat = network.getFlat();
        // copy Propagation from encog, fix flat spot problem
        double[] flatSpot = new double[flat.getActivationFunctions().length];
        for(int i = 0; i < flat.getActivationFunctions().length; i++) {
            flatSpot[i] = flat.getActivationFunctions()[i] instanceof ActivationSigmoid ? 0.1 : 0.0;

        }
        LOG.info("Gradient computing thread count is {}.", modelConfig.getTrain().getWorkerThreadCount());

        this.gradient = new ParallelGradient((FloatFlatNetwork) flat, training, testing, flatSpot,
                new LinearErrorFunction(), isCrossOver, modelConfig.getTrain().getWorkerThreadCount(), this.isELM,
                this.lossStr, this.batchs);
    }

    private NNParams buildEmptyNNParams(WorkerContext<NNParams, NNParams> workerContext) {
        NNParams params = new NNParams();
        params.setWeights(new double[0]);
        params.setGradients(new double[0]);
        params.setTestError(NNConstants.DRY_ERROR);
        params.setTrainError(NNConstants.DRY_ERROR);
        return params;
    }

    @Override
    protected void postLoad(WorkerContext<NNParams, NNParams> workerContext) {
        if(isOnDisk()) {
            ((BufferedFloatMLDataSet) this.trainingData).endLoad();
            if(validationData != null) {
                ((BufferedFloatMLDataSet) this.validationData).endLoad();
            }
        } else {
            ((MemoryDiskFloatMLDataSet) this.trainingData).endLoad();
            ((MemoryDiskFloatMLDataSet) this.validationData).endLoad();
            LOG.info("    - # Training Records in memory: {}.",
                    ((MemoryDiskFloatMLDataSet) this.trainingData).getMemoryCount());
            LOG.info("    - # Training Records in disk: {}.",
                    ((MemoryDiskFloatMLDataSet) this.trainingData).getDiskCount());
        }
        LOG.info("    - # Records of the Total Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        if(this.isKFoldCV) {
            LOG.info("        - Validation Rate(kFold): {}.", 1d / this.modelConfig.getTrain().getNumKFold());
        } else {
            LOG.info("        - Validation Rate: {}.", this.modelConfig.getValidSetRate());
        }
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.getRecordCount());
        if(modelConfig.isRegression() || modelConfig.getTrain().isOneVsAll()) {
            LOG.info("        - # Positive Bagging Selected Records of the Training Set: {}.",
                    this.positiveSelectedTrainCount);
            LOG.info("        - # Negative Bagging Selected Records of the Training Set: {}.",
                    this.negativeSelectedTrainCount);
            LOG.info("        - # Positive Raw Records of the Training Set: {}.", this.positiveTrainCount);
            LOG.info("        - # Negative Raw Records of the Training Set: {}.", this.negativeTrainCount);
        }

        if(validationData != null) {
            LOG.info("        - # Records of the Validation Set: {}.", this.validationData.getRecordCount());
            if(modelConfig.isRegression() || modelConfig.getTrain().isOneVsAll()) {
                LOG.info("        - # Positive Records of the Validation Set: {}.", this.positiveValidationCount);
                LOG.info("        - # Negative Records of the Validation Set: {}.", this.negativeValidationCount);
            }
        }
    }

    protected float sampleWeights(float label) {
        float sampleWeights = 1f;
        // sample negative or kFoldCV, sample rate is 1d
        double sampleRate = (modelConfig.getTrain().getSampleNegOnly() || this.isKFoldCV) ? 1d : modelConfig.getTrain()
                .getBaggingSampleRate();
        int classValue = (int) (label + 0.01f);
        if(!modelConfig.isBaggingWithReplacement()) {
            Random random = null;
            if(this.isStratifiedSampling) {
                random = baggingRandomMap.get(classValue);
                if(random == null) {
                    random = DTrainUtils.generateRandomBySampleSeed(modelConfig.getTrain().getBaggingSampleSeed(),
                            CommonConstants.NOT_CONFIGURED_BAGGING_SEED);
                    baggingRandomMap.put(classValue, random);
                }
            } else {
                random = baggingRandomMap.get(0);
                if(random == null) {
                    random = DTrainUtils.generateRandomBySampleSeed(modelConfig.getTrain().getBaggingSampleSeed(),
                            CommonConstants.NOT_CONFIGURED_BAGGING_SEED);
                    baggingRandomMap.put(0, random);
                }
            }
            if(random.nextDouble() <= sampleRate) {
                sampleWeights = 1f;
            } else {
                sampleWeights = 0f;
            }
        } else {
            // bagging with replacement sampling in training data set, take PoissonDistribution for sampling with
            // replacement
            if(this.isStratifiedSampling) {
                PoissonDistribution rng = this.baggingRngMap.get(classValue);
                if(rng == null) {
                    rng = new PoissonDistribution(sampleRate);
                    this.baggingRngMap.put(classValue, rng);
                }
                sampleWeights = rng.sample();
            } else {
                PoissonDistribution rng = this.baggingRngMap.get(0);
                if(rng == null) {
                    rng = new PoissonDistribution(sampleRate);
                    this.baggingRngMap.put(0, rng);
                }
                sampleWeights = rng.sample();
            }
        }
        return sampleWeights;
    }

    protected void addDataPairToDataSet(long hashcode, FloatMLDataPair pair) {
        addDataPairToDataSet(hashcode, pair, false);
    }

    protected boolean isPositive(float value) {
        return Float.compare(1f, value) == 0 ? true : false;
    }

    /**
     * Add to training set or validation set according to validation rate.
     * 
     * @param hashcode
     *            the hash code of the data
     * @param pair
     *            data instance
     * @param isValidation
     *            if it is validation
     * @return if in training, training is true, others are false.
     */
    protected boolean addDataPairToDataSet(long hashcode, FloatMLDataPair pair, boolean isValidation) {
        if(this.isKFoldCV) {
            int k = this.modelConfig.getTrain().getNumKFold();
            if(hashcode % k == this.trainerId) {
                this.validationData.add(pair);
                if(isPositive(pair.getIdealArray()[0])) {
                    this.positiveValidationCount += 1L;
                } else {
                    this.negativeValidationCount += 1L;
                }
                return false;
            } else {
                this.trainingData.add(pair);
                if(isPositive(pair.getIdealArray()[0])) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        }

        if(this.isSpecificValidation) {
            if(isValidation) {
                this.validationData.add(pair);
                if(isPositive(pair.getIdealArray()[0])) {
                    this.positiveValidationCount += 1L;
                } else {
                    this.negativeValidationCount += 1L;
                }
                return false;
            } else {
                this.trainingData.add(pair);
                if(isPositive(pair.getIdealArray()[0])) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        } else {
            if(Double.compare(this.modelConfig.getValidSetRate(), 0d) != 0) {
                int classValue = (int) (pair.getIdealArray()[0] + 0.01f);
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
                        this.validationData.add(pair);
                        if(isPositive(pair.getIdealArray()[0])) {
                            this.positiveValidationCount += 1L;
                        } else {
                            this.negativeValidationCount += 1L;
                        }
                        return false;
                    } else {
                        this.trainingData.add(pair);
                        if(isPositive(pair.getIdealArray()[0])) {
                            this.positiveTrainCount += 1L;
                        } else {
                            this.negativeTrainCount += 1L;
                        }
                        return true;
                    }
                } else {
                    // not fixed initial input, if random value >= validRate, training, otherwise validation.
                    if(random.nextDouble() >= this.modelConfig.getValidSetRate()) {
                        this.trainingData.add(pair);
                        if(isPositive(pair.getIdealArray()[0])) {
                            this.positiveTrainCount += 1L;
                        } else {
                            this.negativeTrainCount += 1L;
                        }
                        return true;
                    } else {
                        this.validationData.add(pair);
                        if(isPositive(pair.getIdealArray()[0])) {
                            this.positiveValidationCount += 1L;
                        } else {
                            this.negativeValidationCount += 1L;
                        }
                        return false;
                    }
                }
            } else {
                this.trainingData.add(pair);
                if(isPositive(pair.getIdealArray()[0])) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        }
    }

    protected boolean isInRange(long hashcode, int startHashCode, int endHashCode) {
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
     * Only baggingWithReplacement is set and size over NNConstants.NN_BAGGING_THRESHOLD, and random value <= 1/size. We
     * choose use existing data to add training data set and testing data set.
     */
    @SuppressWarnings("unused")
    private boolean isBaggingReplacementTrigged(final double random) {
        long trainingSize = this.trainingData.getRecordCount();
        long testingSize = this.validationData.getRecordCount();
        // size should be equals to sampleCount:)
        long size = trainingSize + testingSize;
        return this.modelConfig.isBaggingWithReplacement() && (testingSize > 0) && (trainingSize > 0)
                && (size > NNConstants.NN_BAGGING_THRESHOLD) && (Double.compare(random, 0.5d) < 0);
    }

    /**
     * From Trainer, the logic is to random choose items in master dataset, but I don't want to load data twice for
     * saving memory. Use this to mock raw random repeat logic. This should be some logic difference because of data are
     * not loaded into data set, not random.
     */
    @SuppressWarnings("unused")
    private void mockRandomRepeatData(double crossValidationRate, double random) {
        long trainingSize = this.trainingData.getRecordCount();
        long testingSize = this.validationData.getRecordCount();
        long size = trainingSize + testingSize;
        // here we used a strong cast from long to int since it's just a random choosing algorithm
        int next = RandomUtils.nextInt((int) size);
        FloatMLDataPair dataPair = new BasicFloatMLDataPair(new BasicFloatMLData(new float[this.subFeatures.size()]),
                new BasicFloatMLData(new float[this.outputNodeCount]));
        if(next >= trainingSize) {
            this.validationData.getRecord(next - trainingSize, dataPair);
        } else {
            this.trainingData.getRecord(next, dataPair);
        }

        if(Double.compare(random, crossValidationRate) < 0) {
            this.validationData.add(dataPair);
        } else {
            this.trainingData.add(dataPair);
        }
    }

    public FloatMLDataSet getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(FloatMLDataSet trainingData) {
        this.trainingData = trainingData;
    }

    public FloatMLDataSet getTestingData() {
        return validationData;
    }

    public void setTestingData(FloatMLDataSet testingData) {
        this.validationData = testingData;
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    public void setModelConfig(ModelConfig modelConfig) {
        this.modelConfig = modelConfig;
    }

    public List<ColumnConfig> getColumnConfigList() {
        return columnConfigList;
    }

    public void setColumnConfigList(List<ColumnConfig> columnConfigList) {
        this.columnConfigList = columnConfigList;
    }

    public int getInputNodeCount() {
        return inputNodeCount;
    }

    public void setInputNodeCount(int inputNodeCount) {
        this.inputNodeCount = inputNodeCount;
    }

    public int getOutputNodeCount() {
        return outputNodeCount;
    }

    public void setOutputNodeCount(int outputNodeCount) {
        this.outputNodeCount = outputNodeCount;
    }

}
