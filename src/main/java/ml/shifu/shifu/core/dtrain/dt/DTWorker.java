/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryLimitedList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.guagua.worker.WorkerContext.WorkerCompletionCallBack;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams.NodeStats;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.ClassUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link DTWorker} is to collection node statistics for node with sub-sampling features from master {@link DTMaster}.
 * 
 * <p>
 * Random forest and gradient boost decision tree are all supported in such worker. For RF, just to collect statistics
 * for nodes from master. For GBDT, extra label and predict updated in each iteration.
 * 
 * <p>
 * For GBDT, loaded data instances will also be changed for predict and label. Which means such data can only be stored
 * into memory. To store predict and label in GBDT, In Data predict and label are all set even with RF. Data are stored
 * as float types to save memory consumption.
 * 
 * <p>
 * For GBDT, when a new tree is transferred to worker, data predict and label are all updated and such value can be
 * covered according to trees and learning rate.
 * 
 * <p>
 * For RF, bagging with replacement are enabled by {@link PoissonDistribution}.
 * 
 * <p>
 * Weighted training are supported in our RF and GBDT impl, in such worker, data.significance is weight field set from
 * input. If no weight, such value is set to 1.
 * 
 * <p>
 * Bin index is stored in each Data object as short to save memory, especially for categorical features, memory is saved
 * a lot from String to short. With short type, number of categories only limited in Short.MAX_VALUE.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 800)
public class DTWorker
        extends
        AbstractWorkerComputable<DTMasterParams, DTWorkerParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(DTWorker.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Total tree numbers
     */
    private int treeNum;

    /**
     * Basic input count for final-select variables or good candidates(if no any variables are selected)
     */
    protected int inputCount;

    /**
     * Basic categorical input count
     */
    protected int categoricalInputCount;

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
     * Training data set with only in memory because for GBDT data will be changed in later iterations.
     */
    private volatile MemoryLimitedList<Data> trainingData;

    /**
     * Validation data set with only in memory because for GBDT data will be changed in later iterations.
     */
    private volatile MemoryLimitedList<Data> validationData;

    /**
     * PoissonDistribution which is used for up sampling positive records.
     */
    protected PoissonDistribution upSampleRng = null;

    /**
     * Bagging with poisson distribution instances
     */
    private Map<Integer, PoissonDistribution[]> baggingRngMap = new HashMap<Integer, PoissonDistribution[]>();

    /**
     * Construct a bagging random map for different classes. For stratified sampling, this is useful for each class
     * sampling.
     */
    private Map<Integer, Random> baggingRandomMap = new HashMap<Integer, Random>();

    /**
     * Construct a validation random map for different classes. For stratified sampling, this is useful for each class
     * sampling.
     */
    private Map<Integer, Random> validationRandomMap = new HashMap<Integer, Random>();

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    protected static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR)
            .trimResults();

    /**
     * Index map in which column index and data input array index for fast location.
     */
    private ConcurrentMap<Integer, Integer> inputIndexMap = new ConcurrentHashMap<Integer, Integer>();

    /**
     * Different {@link Impurity} for node, {@link Entropy} and {@link Gini} are mostly for classification,
     * {@link Variance} are mostly for regression.
     */
    private Impurity impurity;

    /**
     * If for random forest running, this is default for such master.
     */
    private boolean isRF = true;

    /**
     * If gradient boost decision tree, for GBDT, each time a tree is trained, next train is trained by gradient label
     * from previous tree.
     */
    private boolean isGBDT = false;

    /**
     * Learning rate GBDT.
     */
    private double learningRate = 0.1d;

    /**
     * Different loss strategy for GBDT.
     */
    private Loss loss = null;

    /**
     * By default in GBDT, sample with replacement is enabled, but looks sometimes good performance with replacement &
     * GBDT
     */
    private boolean gbdtSampleWithReplacement = false;

    /**
     * Trainer id used to tag bagging training job, starting from 0, 1, 2 ...
     */
    private int trainerId = 0;

    /**
     * If one vs all method for multiple classification.
     */
    private boolean isOneVsAll = false;

    /**
     * Create a thread pool to do gradient computing and test set error computing using multiple threads.
     */
    private ExecutorService threadPool;

    /**
     * Worker thread count used as multiple threading to get node status
     */
    private int workerThreadCount;

    /**
     * Indicates if validation are set by users for validationDataPath, not random picking
     */
    protected boolean isManualValidation = false;

    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled;

    /**
     * Mapping for (ColumnNum, Map(Category, CategoryIndex) for categorical feature
     */
    private Map<Integer, Map<String, Integer>> columnCategoryIndexMapping;

    /**
     * Checkpoint output HDFS file
     */
    private Path checkpointOutput;

    /**
     * Trees for fail over or continous model training, this is recovered from hdfs and no need back up
     */
    private List<TreeNode> recoverTrees;

    /**
     * A flag means current worker is fail over task and gbdt predict value needs to be recovered. After data recovered,
     * such falg should reset to false
     */
    private boolean isNeedRecoverGBDTPredict = false;

    /**
     * If stratified sampling or random sampling
     */
    private boolean isStratifiedSampling = false;

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;

    /**
     * Drop out rate for gbdt to drop trees in training. http://xgboost.readthedocs.io/en/latest/tutorials/dart.html
     */
    private double dropOutRate = 0.0;

    /**
     * Random object to drop out trees, work with {@link #dropOutRate}
     */
    private Random dropOutRandom = new Random(System.currentTimeMillis() + 5000L);

    /**
     * Random object to sample negative records
     */
    private Random sampelNegOnlyRandom = new Random(System.currentTimeMillis() + 1000L);

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    protected boolean isUpSampleEnabled() {
        // only enabled in regression
        return this.upSampleRng != null
                && (modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll()));
    }

    @Override
    public void init(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        Properties props = context.getProps();
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.columnCategoryIndexMapping = new HashMap<Integer, Map<String, Integer>>();
        for(ColumnConfig config: this.columnConfigList) {
            if(config.isCategorical()) {
                if(config.getBinCategory() != null) {
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

        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
            LOG.info("Cross validation is enabled by kCrossValidation: {}.", kCrossValidation);
        }

        Double upSampleWeight = modelConfig.getTrain().getUpSampleWeight();
        if(Double.compare(upSampleWeight, 1d) != 0
                && (modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll()))) {
            // set mean to upSampleWeight -1 and get sample + 1 to make sure no zero sample value
            LOG.info("Enable up sampling with weight {}.", upSampleWeight);
            this.upSampleRng = new PoissonDistribution(upSampleWeight - 1);
        }

        this.isContinuousEnabled = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));

        this.workerThreadCount = modelConfig.getTrain().getWorkerThreadCount();
        this.threadPool = Executors.newFixedThreadPool(this.workerThreadCount);
        // enable shut down logic
        context.addCompletionCallBack(new WorkerCompletionCallBack<DTMasterParams, DTWorkerParams>() {
            @Override
            public void callback(WorkerContext<DTMasterParams, DTWorkerParams> context) {
                DTWorker.this.threadPool.shutdownNow();
                try {
                    DTWorker.this.threadPool.awaitTermination(2, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        this.isOneVsAll = modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll();

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain().getGridConfigFileContent());
        Map<String, Object> validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(this.trainerId);
            LOG.info("Start grid search worker with params: {}", validParams);
        }

        this.treeNum = Integer.valueOf(validParams.get("TreeNum").toString());

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
            if(Double.compare(validationRate, 0d) != 0) {
                this.trainingData = new MemoryLimitedList<Data>((long) (Runtime.getRuntime().maxMemory()
                        * memoryFraction * (1 - validationRate)), new ArrayList<Data>());
                this.validationData = new MemoryLimitedList<Data>((long) (Runtime.getRuntime().maxMemory()
                        * memoryFraction * validationRate), new ArrayList<Data>());
            } else {
                this.trainingData = new MemoryLimitedList<Data>(
                        (long) (Runtime.getRuntime().maxMemory() * memoryFraction), new ArrayList<Data>());
            }
        }

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        // numerical + categorical = # of all input
        this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];
        // regression outputNodeCount is 1, binaryClassfication, it is 1, OneVsAll it is 1, Native classification it is
        // 1, with index of 0,1,2,3 denotes different classes
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        this.isManualValidation = (modelConfig.getValidationDataSetRawPath() != null && !"".equals(modelConfig
                .getValidationDataSetRawPath()));

        int numClasses = this.modelConfig.isClassification() ? this.modelConfig.getTags().size() : 2;
        String imStr = validParams.get("Impurity").toString();
        int minInstancesPerNode = Integer.valueOf(validParams.get("MinInstancesPerNode").toString());
        double minInfoGain = Double.valueOf(validParams.get("MinInfoGain").toString());
        if(imStr.equalsIgnoreCase("entropy")) {
            impurity = new Entropy(numClasses, minInstancesPerNode, minInfoGain);
        } else if(imStr.equalsIgnoreCase("gini")) {
            impurity = new Gini(numClasses, minInstancesPerNode, minInfoGain);
        } else if(imStr.equalsIgnoreCase("friedmanmse")) {
            impurity = new FriedmanMSE(minInstancesPerNode, minInfoGain);
        } else {
            impurity = new Variance(minInstancesPerNode, minInfoGain);
        }

        this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        this.isGBDT = ALGORITHM.GBT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());

        String lossStr = validParams.get("Loss").toString();
        if(lossStr.equalsIgnoreCase("log")) {
            this.loss = new LogLoss();
        } else if(lossStr.equalsIgnoreCase("absolute")) {
            this.loss = new AbsoluteLoss();
        } else if(lossStr.equalsIgnoreCase("halfgradsquared")) {
            this.loss = new HalfGradSquaredLoss();
        } else if(lossStr.equalsIgnoreCase("squared")) {
            this.loss = new SquaredLoss();
        } else {
            try {
                this.loss = (Loss) ClassUtils.newInstance(Class.forName(lossStr));
            } catch (ClassNotFoundException e) {
                LOG.warn("Class not found for {}, using default SquaredLoss", lossStr);
                this.loss = new SquaredLoss();
            }
        }

        if(this.isGBDT) {
            this.learningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());
            Object swrObj = validParams.get("GBTSampleWithReplacement");
            if(swrObj != null) {
                this.gbdtSampleWithReplacement = Boolean.TRUE.toString().equalsIgnoreCase(swrObj.toString());
            }

            Object dropoutObj = validParams.get(CommonConstants.DROPOUT_RATE);
            if(dropoutObj != null) {
                this.dropOutRate = Double.valueOf(dropoutObj.toString());
            }
        }

        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();

        this.checkpointOutput = new Path(context.getProps().getProperty(
                CommonConstants.SHIFU_DT_MASTER_CHECKPOINT_FOLDER, "tmp/cp_" + context.getAppId()));

        LOG.info(
                "Worker init params:isAfterVarSel={}, treeNum={}, impurity={}, loss={}, learningRate={}, gbdtSampleWithReplacement={}, isRF={}, isGBDT={}, isStratifiedSampling={}, isKFoldCV={}, kCrossValidation={}, dropOutRate={}",
                isAfterVarSelect, treeNum, impurity.getClass().getName(), loss.getClass().getName(), this.learningRate,
                this.gbdtSampleWithReplacement, this.isRF, this.isGBDT, this.isStratifiedSampling, this.isKFoldCV,
                kCrossValidation, this.dropOutRate);

        // for fail over, load existing trees
        if(!context.isFirstIteration()) {
            if(this.isGBDT) {
                // set flag here and recover later in doComputing, this is to make sure recover after load part which
                // can load latest trees in #doCompute
                isNeedRecoverGBDTPredict = true;
            } else {
                // RF , trees are recovered from last master results
                recoverTrees = context.getLastMasterResult().getTrees();
            }
        }

        if(context.isFirstIteration() && this.isContinuousEnabled && this.isGBDT) {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            TreeModel existingModel = null;
            try {
                existingModel = (TreeModel) CommonUtils.loadModel(modelConfig, modelPath,
                        ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource()));
            } catch (IOException e) {
                LOG.error("Error in get existing model, will ignore and start from scratch", e);
            }
            if(existingModel == null) {
                LOG.warn("No mdel is found even set to continuous model training.");
                return;
            } else {
                recoverTrees = existingModel.getTrees();
                LOG.info("Loading existing {} trees", recoverTrees.size());
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#doCompute(ml.shifu.guagua.worker.WorkerContext)
     */
    @Override
    public DTWorkerParams doCompute(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        if(context.isFirstIteration()) {
            return new DTWorkerParams();
        }

        DTMasterParams lastMasterResult = context.getLastMasterResult();
        final List<TreeNode> trees = lastMasterResult.getTrees();
        final Map<Integer, TreeNode> todoNodes = lastMasterResult.getTodoNodes();
        if(todoNodes == null) {
            return new DTWorkerParams();
        }

        Map<Integer, NodeStats> statistics = initTodoNodeStats(todoNodes);

        double trainError = 0d, validationError = 0d;
        double weightedTrainCount = 0d, weightedValidationCount = 0d;
        // renew random seed
        if(this.isGBDT && !this.gbdtSampleWithReplacement && lastMasterResult.isSwitchToNextTree()) {
            this.baggingRandomMap = new HashMap<Integer, Random>();
        }

        long start = System.nanoTime();
        for(Data data: this.trainingData) {
            if(this.isRF) {
                for(TreeNode treeNode: trees) {
                    if(treeNode.getNode().getId() == Node.INVALID_INDEX) {
                        continue;
                    }

                    Node predictNode = predictNodeIndex(treeNode.getNode(), data, true);
                    if(predictNode.getPredict() != null) {
                        // only update when not in first node, for treeNode, no predict statistics at that time
                        float weight = data.subsampleWeights[treeNode.getTreeId()];
                        if(Float.compare(weight, 0f) == 0) {
                            // oob data, no need to do weighting
                            validationError += data.significance
                                    * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                            weightedValidationCount += data.significance;
                        } else {
                            trainError += weight * data.significance
                                    * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                            weightedTrainCount += weight * data.significance;
                        }
                    }
                }
            }

            if(this.isGBDT) {
                if(this.isContinuousEnabled && lastMasterResult.isContinuousRunningStart()) {
                    recoverGBTData(context, data.output, data.predict, data, false);
                    trainError += data.significance * loss.computeError(data.predict, data.label);
                    weightedTrainCount += data.significance;
                } else {
                    if(isNeedRecoverGBDTPredict) {
                        if(this.recoverTrees == null) {
                            this.recoverTrees = recoverCurrentTrees();
                        }
                        // recover gbdt data for fail over
                        recoverGBTData(context, data.output, data.predict, data, true);
                    }
                    int currTreeIndex = trees.size() - 1;

                    if(lastMasterResult.isSwitchToNextTree()) {
                        if(currTreeIndex >= 1) {
                            Node node = trees.get(currTreeIndex - 1).getNode();
                            Node predictNode = predictNodeIndex(node, data, false);
                            if(predictNode.getPredict() != null) {
                                double predict = predictNode.getPredict().getPredict();
                                // first tree logic, master must set it to first tree even second tree with ROOT is
                                // sending
                                if(context.getLastMasterResult().isFirstTree()) {
                                    data.predict = (float) predict;
                                } else {
                                    // random drop
                                    boolean drop = (this.dropOutRate > 0.0 && dropOutRandom.nextDouble() < this.dropOutRate);
                                    if(!drop) {
                                        data.predict += (float) (this.learningRate * predict);
                                    }
                                }
                                data.output = -1f * loss.computeGradient(data.predict, data.label);
                            }
                            // if not sampling with replacement in gbdt, renew bagging sample rate in next tree
                            if(!this.gbdtSampleWithReplacement) {
                                Random random = null;
                                int classValue = (int) (data.label + 0.01f);
                                if(this.isStratifiedSampling) {
                                    random = baggingRandomMap.get(classValue);
                                    if(random == null) {
                                        random = DTrainUtils.generateRandomBySampleSeed(
                                                modelConfig.getTrain().getBaggingSampleSeed(),
                                                CommonConstants.NOT_CONFIGURED_BAGGING_SEED);
                                        baggingRandomMap.put(classValue, random);
                                    }
                                } else {
                                    random = baggingRandomMap.get(0);
                                    if(random == null) {
                                        random = DTrainUtils.generateRandomBySampleSeed(
                                                modelConfig.getTrain().getBaggingSampleSeed(),
                                                CommonConstants.NOT_CONFIGURED_BAGGING_SEED);
                                        baggingRandomMap.put(0, random);
                                    }
                                }
                                if(random.nextDouble() <= modelConfig.getTrain().getBaggingSampleRate()) {
                                    data.subsampleWeights[currTreeIndex % data.subsampleWeights.length] = 1f;
                                } else {
                                    data.subsampleWeights[currTreeIndex % data.subsampleWeights.length] = 0f;
                                }
                            }
                        }
                    }

                    if(context.getLastMasterResult().isFirstTree() && !lastMasterResult.isSwitchToNextTree()) {
                        Node currTree = trees.get(currTreeIndex).getNode();
                        Node predictNode = predictNodeIndex(currTree, data, true);
                        if(predictNode.getPredict() != null) {
                            trainError += data.significance
                                    * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                            weightedTrainCount += data.significance;
                        }
                    } else {
                        trainError += data.significance * loss.computeError(data.predict, data.label);
                        weightedTrainCount += data.significance;
                    }
                }
            }
        }
        LOG.debug("Compute train error time is {}ms", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));

        if(validationData != null) {
            start = System.nanoTime();
            for(Data data: this.validationData) {
                if(this.isRF) {
                    for(TreeNode treeNode: trees) {
                        if(treeNode.getNode().getId() == Node.INVALID_INDEX) {
                            continue;
                        }
                        Node predictNode = predictNodeIndex(treeNode.getNode(), data, true);
                        if(predictNode.getPredict() != null) {
                            // only update when not in first node, for treeNode, no predict statistics at that time
                            validationError += data.significance
                                    * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                            weightedValidationCount += data.significance;
                        }
                    }
                }

                if(this.isGBDT) {
                    if(this.isContinuousEnabled && lastMasterResult.isContinuousRunningStart()) {
                        recoverGBTData(context, data.output, data.predict, data, false);
                        validationError += data.significance * loss.computeError(data.predict, data.label);
                        weightedValidationCount += data.significance;
                    } else {
                        if(isNeedRecoverGBDTPredict) {
                            if(this.recoverTrees == null) {
                                this.recoverTrees = recoverCurrentTrees();
                            }
                            // recover gbdt data for fail over
                            recoverGBTData(context, data.output, data.predict, data, true);
                        }
                        int currTreeIndex = trees.size() - 1;
                        if(lastMasterResult.isSwitchToNextTree()) {
                            if(currTreeIndex >= 1) {
                                Node node = trees.get(currTreeIndex - 1).getNode();
                                Node predictNode = predictNodeIndex(node, data, false);
                                if(predictNode.getPredict() != null) {
                                    double predict = predictNode.getPredict().getPredict();
                                    if(context.getLastMasterResult().isFirstTree()) {
                                        data.predict = (float) predict;
                                    } else {
                                        data.predict += (float) (this.learningRate * predict);
                                    }
                                    data.output = -1f * loss.computeGradient(data.predict, data.label);
                                }
                            }
                        }
                        if(context.getLastMasterResult().isFirstTree() && !lastMasterResult.isSwitchToNextTree()) {
                            Node predictNode = predictNodeIndex(trees.get(currTreeIndex).getNode(), data, true);
                            if(predictNode.getPredict() != null) {
                                validationError += data.significance
                                        * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                                weightedValidationCount += data.significance;
                            }
                        } else {
                            validationError += data.significance * loss.computeError(data.predict, data.label);
                            weightedValidationCount += data.significance;
                        }
                    }
                }
            }
            LOG.debug("Compute val error time is {}ms", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));
        }

        if(this.isGBDT) {
            // reset trees to null to save memory
            this.recoverTrees = null;
            if(this.isNeedRecoverGBDTPredict) {
                // no need recover again
                this.isNeedRecoverGBDTPredict = false;
            }
        }

        start = System.nanoTime();
        CompletionService<Map<Integer, NodeStats>> completionService = new ExecutorCompletionService<Map<Integer, NodeStats>>(
                this.threadPool);

        int realThreadCount = 0;
        LOG.debug("while todo size {}", todoNodes.size());

        int realRecords = this.trainingData.size();
        int realThreads = this.workerThreadCount > realRecords ? realRecords : this.workerThreadCount;

        int[] trainLows = new int[realThreads];
        int[] trainHighs = new int[realThreads];

        int stepCount = realRecords / realThreads;
        if(realRecords % realThreads != 0) {
            // move step count to append last gap to avoid last thread worse 2*stepCount-1
            stepCount += (realRecords % realThreads) / stepCount;
        }
        for(int i = 0; i < realThreads; i++) {
            trainLows[i] = i * stepCount;
            if(i != realThreads - 1) {
                trainHighs[i] = trainLows[i] + stepCount - 1;
            } else {
                trainHighs[i] = realRecords - 1;
            }
        }

        for(int i = 0; i < realThreads; i++) {
            final Map<Integer, TreeNode> localTodoNodes = new HashMap<Integer, TreeNode>(todoNodes);
            final Map<Integer, NodeStats> localStatistics = initTodoNodeStats(todoNodes);

            final int startIndex = trainLows[i];
            final int endIndex = trainHighs[i];
            LOG.info("Thread {} todo size {} stats size {} start index {} end index {}", i, localTodoNodes.size(),
                    localStatistics.size(), startIndex, endIndex);

            if(localTodoNodes.size() == 0) {
                continue;
            }
            realThreadCount += 1;
            completionService.submit(new Callable<Map<Integer, NodeStats>>() {
                @Override
                public Map<Integer, NodeStats> call() throws Exception {
                    long start = System.nanoTime();
                    List<Integer> nodeIndexes = new ArrayList<Integer>(trees.size());
                    for(int j = startIndex; j <= endIndex; j++) {
                        Data data = DTWorker.this.trainingData.get(j);
                        nodeIndexes.clear();
                        if(DTWorker.this.isRF) {
                            for(TreeNode treeNode: trees) {
                                if(treeNode.getNode().getId() == Node.INVALID_INDEX) {
                                    nodeIndexes.add(Node.INVALID_INDEX);
                                } else {
                                    Node predictNode = predictNodeIndex(treeNode.getNode(), data, false);
                                    nodeIndexes.add(predictNode.getId());
                                }
                            }
                        }

                        if(DTWorker.this.isGBDT) {
                            int currTreeIndex = trees.size() - 1;
                            Node predictNode = predictNodeIndex(trees.get(currTreeIndex).getNode(), data, false);
                            // update node index
                            nodeIndexes.add(predictNode.getId());
                        }
                        for(Map.Entry<Integer, TreeNode> entry: localTodoNodes.entrySet()) {
                            // only do statistics on effective data
                            Node todoNode = entry.getValue().getNode();
                            int treeId = entry.getValue().getTreeId();
                            int currPredictIndex = 0;
                            if(DTWorker.this.isRF) {
                                currPredictIndex = nodeIndexes.get(entry.getValue().getTreeId());
                            }
                            if(DTWorker.this.isGBDT) {
                                currPredictIndex = nodeIndexes.get(0);
                            }

                            if(todoNode.getId() == currPredictIndex) {
                                List<Integer> features = entry.getValue().getFeatures();
                                if(features.isEmpty()) {
                                    features = getAllValidFeatures();
                                }
                                for(Integer columnNum: features) {
                                    double[] featuerStatistic = localStatistics.get(entry.getKey())
                                            .getFeatureStatistics().get(columnNum);
                                    float weight = data.subsampleWeights[treeId % data.subsampleWeights.length];
                                    if(Float.compare(weight, 0f) != 0) {
                                        // only compute weight is not 0
                                        short binIndex = data.inputs[DTWorker.this.inputIndexMap.get(columnNum)];
                                        DTWorker.this.impurity.featureUpdate(featuerStatistic, binIndex, data.output,
                                                data.significance, weight);
                                    }
                                }
                            }
                        }
                    }
                    LOG.debug("Thread computing stats time is {}ms in thread {}",
                            TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start), Thread.currentThread().getName());
                    return localStatistics;
                }
            });
        }

        int rCnt = 0;
        while(rCnt < realThreadCount) {
            try {
                Map<Integer, NodeStats> currNodeStatsmap = completionService.take().get();
                if(rCnt == 0) {
                    statistics = currNodeStatsmap;
                } else {
                    for(Entry<Integer, NodeStats> entry: statistics.entrySet()) {
                        NodeStats resultNodeStats = entry.getValue();
                        mergeNodeStats(resultNodeStats, currNodeStatsmap.get(entry.getKey()));
                    }
                }
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            rCnt += 1;
        }
        LOG.debug("Compute stats time is {}ms", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));

        LOG.info(
                "worker count is {}, error is {}, and stats size is {}. weightedTrainCount {}, weightedValidationCount {}, trainError {}, validationError {}",
                count, trainError, statistics.size(), weightedTrainCount, weightedValidationCount, trainError,
                validationError);
        return new DTWorkerParams(weightedTrainCount, weightedValidationCount, trainError, validationError, statistics);
    }

    private void mergeNodeStats(NodeStats resultNodeStats, NodeStats nodeStats) {
        Map<Integer, double[]> featureStatistics = resultNodeStats.getFeatureStatistics();
        for(Entry<Integer, double[]> entry: nodeStats.getFeatureStatistics().entrySet()) {
            double[] statistics = featureStatistics.get(entry.getKey());
            for(int i = 0; i < statistics.length; i++) {
                statistics[i] += entry.getValue()[i];
            }
        }
    }

    private Map<Integer, NodeStats> initTodoNodeStats(Map<Integer, TreeNode> todoNodes) {
        Map<Integer, NodeStats> statistics = new HashMap<Integer, NodeStats>(todoNodes.size(), 1f);
        for(Map.Entry<Integer, TreeNode> entry: todoNodes.entrySet()) {
            List<Integer> features = entry.getValue().getFeatures();
            if(features.isEmpty()) {
                features = getAllValidFeatures();
            }
            Map<Integer, double[]> featureStatistics = new HashMap<Integer, double[]>(features.size(), 1f);
            for(Integer columnNum: features) {
                ColumnConfig columnConfig = this.columnConfigList.get(columnNum);
                if(columnConfig.isNumerical()) {
                    // TODO, how to process null bin
                    int featureStatsSize = columnConfig.getBinBoundary().size() * this.impurity.getStatsSize();
                    featureStatistics.put(columnNum, new double[featureStatsSize]);
                } else if(columnConfig.isCategorical()) {
                    // the last one is for invalid value category like ?, *, ...
                    int featureStatsSize = (columnConfig.getBinCategory().size() + 1) * this.impurity.getStatsSize();
                    featureStatistics.put(columnNum, new double[featureStatsSize]);
                }
            }
            NodeStats nodeStats = new NodeStats(entry.getValue().getTreeId(), entry.getValue().getNode().getId(),
                    featureStatistics);

            statistics.put(entry.getKey(), nodeStats);
        }
        return statistics;
    }

    @Override
    protected void postLoad(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        // need to switch state for read
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

    private List<Integer> getAllValidFeatures() {
        List<Integer> features = new ArrayList<Integer>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    // only select numerical feature with getBinBoundary().size() larger than 1
                    // or categorical feature with getBinCategory().size() larger than 0
                    if((config.isNumerical() && config.getBinBoundary().size() > 1)
                            || (config.isCategorical() && config.getBinCategory().size() > 0)) {
                        features.add(config.getColumnNum());
                    }
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config, hasCandidates)) {
                    // only select numerical feature with getBinBoundary().size() larger than 1
                    // or categorical feature with getBinCategory().size() larger than 0
                    if((config.isNumerical() && config.getBinBoundary().size() > 1)
                            || (config.isCategorical() && config.getBinCategory().size() > 0)) {
                        features.add(config.getColumnNum());
                    }
                }
            }
        }
        return features;
    }

    /**
     * 'binBoundary' is ArrayList in fact, so we can use get method. ["-Infinity", 1d, 4d, ....]
     * 
     * @param value
     *            the value to be checked
     * @param binBoundary
     *            the bin boundary list
     * @return the index in which bin
     */
    public static int getBinIndex(float value, List<Double> binBoundary) {
        if(binBoundary.size() <= 1) {
            // feature with binBoundary.size() <= 1 will not be send to worker, while such feature is still loading into
            // memory, just return the first bin index to avoid exception, while actually such feature isn't used in
            // GBT/RF.
            return 0;
        }

        // the last bin if positive infinity
        if(value == Float.POSITIVE_INFINITY) {
            return binBoundary.size() - 1;
        }

        // the first bin if negative infinity
        if(value == Float.NEGATIVE_INFINITY) {
            return 0;
        }

        int low = 0, high = binBoundary.size() - 1;
        while(low <= high) {
            int mid = (low + high) >>> 1;
            double lowThreshold = binBoundary.get(mid);
            double highThreshold = mid == binBoundary.size() - 1 ? Double.MAX_VALUE : binBoundary.get(mid + 1);
            if(value >= lowThreshold && value < highThreshold) {
                return mid;
            }
            if(value >= highThreshold) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return -1;
    }

    private Node predictNodeIndex(Node node, Data data, boolean isForErr) {
        Node currNode = node;
        Split split = currNode.getSplit();

        // if is leaf
        if(split == null || (currNode.getLeft() == null && currNode.getRight() == null)) {
            return currNode;
        }

        ColumnConfig columnConfig = this.columnConfigList.get(split.getColumnNum());

        Node nextNode = null;
        Integer inputIndex = this.inputIndexMap.get(split.getColumnNum());
        if(inputIndex == null) {
            throw new IllegalStateException("InputIndex should not be null: Split is " + split + ", inputIndexMap is "
                    + this.inputIndexMap + ", data is " + data);
        }
        short value = 0;
        if(columnConfig.isNumerical()) {
            short binIndex = data.inputs[inputIndex];
            value = binIndex;
            double valueToBinLowestValue = columnConfig.getBinBoundary().get(binIndex);
            if(valueToBinLowestValue < split.getThreshold()) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        } else if(columnConfig.isCategorical()) {
            short indexValue = (short) (columnConfig.getBinCategory().size());
            value = indexValue;
            if(data.inputs[inputIndex] >= 0 && data.inputs[inputIndex] < (short) (columnConfig.getBinCategory().size())) {
                indexValue = data.inputs[inputIndex];
            } else {
                // for invalid category, set to last one
                indexValue = (short) (columnConfig.getBinCategory().size());
            }
            if(split.getLeftOrRightCategories().contains(indexValue)) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }

            Set<Short> childCategories = split.getLeftOrRightCategories();
            if(split.isLeft()) {
                if(childCategories.contains(indexValue)) {
                    nextNode = currNode.getLeft();
                } else {
                    nextNode = currNode.getRight();
                }
            } else {
                if(childCategories.contains(indexValue)) {
                    nextNode = currNode.getRight();
                } else {
                    nextNode = currNode.getLeft();
                }
            }
        }

        if(nextNode == null) {
            throw new IllegalStateException("NextNode is null, parent id is " + currNode.getId() + "; parent split is "
                    + split + "; left is " + currNode.getLeft() + "; right is " + currNode.getRight() + "; value is "
                    + value);
        }
        return predictNodeIndex(nextNode, data, isForErr);
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<DTMasterParams, DTWorkerParams> context) {
        this.count += 1;
        if((this.count) % 5000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        // hashcode for fixed input split in train and validation
        long hashcode = 0;

        short[] inputs = new short[this.inputCount];
        float ideal = 0f;
        float significance = 1f;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, inputIndex = 0;
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            if(index == this.columnConfigList.size()) {
                // do we need to check if not weighted directly set to 1f; if such logic non-weight at first, then
                // weight, how to process???
                if(StringUtils.isBlank(modelConfig.getWeightColumnName())) {
                    significance = 1f;
                    break;
                }
                // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 1f)
                significance = input.length() == 0 ? 1f : NumberFormatUtils.getFloat(input, 1f);
                // if invalid weight, set it to 1f and warning in log
                if(Float.compare(significance, 0f) < 0) {
                    LOG.warn("The {} record in current worker weight {} is less than 0f, it is invalid, set it to 1.",
                            count, significance);
                    significance = 1f;
                }
                // the last field is significance, break here
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    ideal = getFloatValue(input);
                } else {
                    if(!isAfterVarSelect) {
                        // no variable selected, good candidate but not meta and not target chose
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                            if(columnConfig.isNumerical()) {
                                float floatValue = getFloatValue(input);
                                // cast is safe as we limit max bin to Short.MAX_VALUE
                                short binIndex = (short) getBinIndex(floatValue, columnConfig.getBinBoundary());
                                inputs[inputIndex] = binIndex;
                                if(!this.inputIndexMap.containsKey(columnConfig.getColumnNum())) {
                                    this.inputIndexMap.put(columnConfig.getColumnNum(), inputIndex);
                                }
                            } else if(columnConfig.isCategorical()) {
                                short shortValue = (short) (columnConfig.getBinCategory().size());
                                if(input.length() == 0) {
                                    // empty
                                    shortValue = (short) (columnConfig.getBinCategory().size());
                                } else {
                                    Integer categoricalIndex = this.columnCategoryIndexMapping.get(
                                            columnConfig.getColumnNum()).get(input);
                                    if(categoricalIndex == null) {
                                        shortValue = -1; // invalid category, set to -1 for last index
                                    } else {
                                        // cast is safe as we limit max bin to Short.MAX_VALUE
                                        shortValue = (short) (categoricalIndex.intValue());
                                    }
                                    if(shortValue == -1) {
                                        // not found
                                        shortValue = (short) (columnConfig.getBinCategory().size());
                                    }
                                }
                                inputs[inputIndex] = shortValue;
                                if(!this.inputIndexMap.containsKey(columnConfig.getColumnNum())) {
                                    this.inputIndexMap.put(columnConfig.getColumnNum(), inputIndex);
                                }
                            }
                            hashcode = hashcode * 31 + input.hashCode();
                            inputIndex += 1;
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            if(columnConfig.isNumerical()) {
                                float floatValue = getFloatValue(input);
                                // cast is safe as we limit max bin to Short.MAX_VALUE
                                short binIndex = (short) getBinIndex(floatValue, columnConfig.getBinBoundary());
                                inputs[inputIndex] = binIndex;
                                if(!this.inputIndexMap.containsKey(columnConfig.getColumnNum())) {
                                    this.inputIndexMap.put(columnConfig.getColumnNum(), inputIndex);
                                }
                            } else if(columnConfig.isCategorical()) {
                                // cast is safe as we limit max bin to Short.MAX_VALUE
                                short shortValue = (short) (columnConfig.getBinCategory().size());
                                if(input.length() == 0) {
                                    // empty
                                    shortValue = (short) (columnConfig.getBinCategory().size());
                                } else {
                                    Integer categoricalIndex = this.columnCategoryIndexMapping.get(
                                            columnConfig.getColumnNum()).get(input);
                                    if(categoricalIndex == null) {
                                        shortValue = -1; // invalid category, set to -1 for last index
                                    } else {
                                        // cast is safe as we limit max bin to Short.MAX_VALUE
                                        shortValue = (short) (categoricalIndex.intValue());
                                    }
                                    if(shortValue == -1) {
                                        // not found
                                        shortValue = (short) (columnConfig.getBinCategory().size());
                                    }
                                }
                                inputs[inputIndex] = shortValue;
                                if(!this.inputIndexMap.containsKey(columnConfig.getColumnNum())) {
                                    this.inputIndexMap.put(columnConfig.getColumnNum(), inputIndex);
                                }
                            }
                            hashcode = hashcode * 31 + input.hashCode();
                            inputIndex += 1;
                        }
                    }
                }
            }
            index += 1;
        }

        if(this.isOneVsAll) {
            // if one vs all, update target value according to index of target
            ideal = updateOneVsAllTargetValue(ideal);
        }

        // sample negative only logic here
        if(modelConfig.getTrain().getSampleNegOnly()) {
            if(this.modelConfig.isFixInitialInput()) {
                // if fixInitialInput, sample hashcode in 1-sampleRate range out if negative records
                int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                // here BaggingSampleRate means how many data will be used in training and validation, if it is 0.8, we
                // should take 1-0.8 to check endHashCode
                int endHashCode = startHashCode
                        + Double.valueOf((1d - this.modelConfig.getBaggingSampleRate()) * 100).intValue();
                if((modelConfig.isRegression() || this.isOneVsAll) // regression or onevsall
                        && (int) (ideal + 0.01d) == 0 // negative record
                        && isInRange(hashcode, startHashCode, endHashCode)) {
                    return;
                }
            } else {
                // if not fixed initial input, and for regression or onevsall multiple classification (regression also).
                // and if negative record do sampling out
                if((modelConfig.isRegression() || this.isOneVsAll) // regression or onevsall
                        && (int) (ideal + 0.01d) == 0 // negative record
                        && Double.compare(this.sampelNegOnlyRandom.nextDouble(),
                                this.modelConfig.getBaggingSampleRate()) >= 0) {
                    return;
                }
            }
        }

        float output = ideal;
        float predict = ideal;

        // up sampling logic, just add more weights while bagging sampling rate is still not changed
        if(modelConfig.isRegression() && isUpSampleEnabled() && Double.compare(ideal, 1d) == 0) {
            // Double.compare(ideal, 1d) == 0 means positive tags; sample + 1 to avoid sample count to 0
            significance = significance * (this.upSampleRng.sample() + 1);
        }

        Data data = new Data(inputs, predict, output, output, significance);

        boolean isValidation = false;
        if(context.getAttachment() != null && context.getAttachment() instanceof Boolean) {
            isValidation = (Boolean) context.getAttachment();
        }

        // split into validation and training data set according to validation rate
        boolean isInTraining = this.addDataPairToDataSet(hashcode, data, isValidation);

        // do bagging sampling only for training data
        if(isInTraining) {
            data.subsampleWeights = sampleWeights(data.label);
            // for training data, compute real selected training data according to baggingSampleRate
            // if gbdt, only the 1st sampling value is used, if rf, use the 1st to denote some information, no need all
            if(isPositive(data.label)) {
                this.positiveSelectedTrainCount += data.subsampleWeights[0] * 1L;
            } else {
                this.negativeSelectedTrainCount += data.subsampleWeights[0] * 1L;
            }
        } else {
            // for validation data, according bagging sampling logic, we may need to sampling validation data set, while
            // validation data set are only used to compute validation error, not to do real sampling is ok.
        }
    }

    private float getFloatValue(String input) {
        // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
        float floatValue = input.length() == 0 ? 0f : NumberFormatUtils.getFloat(input, 0f);
        // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
        floatValue = (Float.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;
        return floatValue;
    }

    private boolean isPositive(float value) {
        return Float.compare(1f, value) == 0 ? true : false;
    }

    /**
     * Add to training set or validation set according to validation rate.
     * 
     * @param hashcode
     *            the hash code of the data
     * @param data
     *            data instance
     * @param isValidation
     *            if it is validation
     * @return if in training, training is true, others are false.
     */
    protected boolean addDataPairToDataSet(long hashcode, Data data, boolean isValidation) {
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

    // isFailoverOrContinuous true failover task, isFailoverOrContinuous false continuous model training
    private void recoverGBTData(WorkerContext<DTMasterParams, DTWorkerParams> context, float output, float predict,
            Data data, boolean isFailoverOrContinuous) {
        final List<TreeNode> trees = this.recoverTrees;
        if(trees == null) {
            return;
        }

        if(trees.size() >= 1) {
            // if isSwitchToNextTree == false, iterate all trees except current one to get new predict and
            // output value; if isSwitchToNextTree == true, iterate all trees except current two trees.
            // the last tree is a root node, the tree with index size-2 will be called in doCompute method
            // TreeNode lastTree = trees.get(trees.size() - 1);

            // if is fail over and trees size over 2, exclude last tree because last tree isn't built full and no need
            // to update predict value, if for continuous model training, all trees are good and should be finished
            // updating predict
            int iterLen = isFailoverOrContinuous ? trees.size() - 1 : trees.size();
            for(int i = 0; i < iterLen; i++) {
                TreeNode currTree = trees.get(i);
                if(i == 0) {
                    double oldPredict = predictNodeIndex(currTree.getNode(), data, false).getPredict().getPredict();
                    predict = (float) oldPredict;
                    output = -1f * loss.computeGradient(predict, data.label);
                } else {
                    // random drop
                    if(this.dropOutRate > 0.0 && dropOutRandom.nextDouble() < this.dropOutRate) {
                        continue;
                    }
                    double oldPredict = predictNodeIndex(currTree.getNode(), data, false).getPredict().getPredict();
                    predict += (float) (this.learningRate * oldPredict);
                    output = -1f * loss.computeGradient(predict, data.label);
                }
            }
            data.output = output;
            data.predict = predict;
        }
    }

    private List<TreeNode> recoverCurrentTrees() {
        FSDataInputStream stream = null;
        List<TreeNode> trees = null;
        try {
            if(!ShifuFileUtils
                    .isFileExists(this.checkpointOutput.toString(), this.modelConfig.getDataSet().getSource())) {
                return null;
            }
            FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource());
            stream = fs.open(this.checkpointOutput);
            int treeSize = stream.readInt();
            trees = new ArrayList<TreeNode>(treeSize);
            for(int i = 0; i < treeSize; i++) {
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(stream);
                trees.add(treeNode);
            }
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        } finally {
            org.apache.commons.io.IOUtils.closeQuietly(stream);
        }
        return trees;
    }

    private float[] sampleWeights(float label) {
        float[] sampleWeights = null;
        // sample negative or kFoldCV, sample rate is 1d
        double sampleRate = (modelConfig.getTrain().getSampleNegOnly() || this.isKFoldCV) ? 1d : modelConfig.getTrain()
                .getBaggingSampleRate();
        int classValue = (int) (label + 0.01f);
        if(this.treeNum == 1 || (this.isGBDT && !this.gbdtSampleWithReplacement)) {
            // if tree == 1 or GBDT, don't use with replacement sampling; for GBDT, every time is one tree
            sampleWeights = new float[1];
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
                sampleWeights[0] = 1f;
            } else {
                sampleWeights[0] = 0f;
            }
        } else {
            // if gbdt and gbdtSampleWithReplacement = true, still sampling with replacement
            sampleWeights = new float[this.treeNum];
            if(this.isStratifiedSampling) {
                PoissonDistribution[] rng = this.baggingRngMap.get(classValue);
                if(rng == null) {
                    rng = new PoissonDistribution[treeNum];
                    for(int i = 0; i < treeNum; i++) {
                        rng[i] = new PoissonDistribution(sampleRate);
                    }
                    this.baggingRngMap.put(classValue, rng);
                }
                for(int i = 0; i < sampleWeights.length; i++) {
                    sampleWeights[i] = rng[i].sample();
                }
            } else {
                PoissonDistribution[] rng = this.baggingRngMap.get(0);
                if(rng == null) {
                    rng = new PoissonDistribution[treeNum];
                    for(int i = 0; i < treeNum; i++) {
                        rng[i] = new PoissonDistribution(sampleRate);
                    }
                    this.baggingRngMap.put(0, rng);
                }
                for(int i = 0; i < sampleWeights.length; i++) {
                    sampleWeights[i] = rng[i].sample();
                }
            }
        }
        return sampleWeights;
    }

    private float updateOneVsAllTargetValue(float ideal) {
        // if one vs all, set correlated idea value according to trainerId which means in trainer with id 0, target
        // 0 is treated with 1, other are 0. Such target value are set to index of tags like [0, 1, 2, 3] compared
        // with ["a", "b", "c", "d"]
        return Float.compare(ideal, trainerId) == 0 ? 1f : 0f;
    }

    static class Data implements Serializable, Bytable {

        private static final long serialVersionUID = 903201066309036170L;

        /**
         * Inputs for bin index, short is using to compress data, for numerical, it can be byte type for less than 256
         * bins, while it is hard to control byte and short together, as so far memory consumption is OK, just use short
         * for both numerical and categorical columns
         */
        short[] inputs;

        /**
         * Original output label and not changed in GBDT
         */
        float label;

        /**
         * Output label and maybe changed in GBDT
         */
        volatile float output;
        volatile float predict;
        float significance;
        float[] subsampleWeights = new float[] { 1.0f };

        public Data() {
            this.label = 0;
        }

        public Data(short[] inputs, float predict, float output, float label, float significance) {
            this.inputs = inputs;
            this.predict = predict;
            this.output = output;
            this.label = label;
            this.significance = significance;
        }

        public Data(short[] inputs, float predict, float output, float label, float significance,
                float[] subsampleWeights) {
            this.inputs = inputs;
            this.predict = predict;
            this.output = output;
            this.label = label;
            this.significance = significance;
            this.subsampleWeights = subsampleWeights;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(inputs.length);
            for(short input: inputs) {
                out.writeShort(input);
            }

            out.writeFloat(output);
            out.writeFloat(label);
            out.writeFloat(predict);

            out.writeFloat(significance);

            out.writeInt(subsampleWeights.length);
            for(float sample: subsampleWeights) {
                out.writeFloat(sample);
            }
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            int iLen = in.readInt();
            this.inputs = new short[iLen];
            for(int i = 0; i < iLen; i++) {
                this.inputs[i] = in.readShort();
            }

            this.output = in.readFloat();
            this.label = in.readFloat();
            this.predict = in.readFloat();

            this.significance = in.readFloat();

            int sLen = in.readInt();
            this.subsampleWeights = new float[sLen];
            for(int i = 0; i < sLen; i++) {
                this.subsampleWeights[i] = in.readFloat();
            }
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#toString()
         */
        @Override
        public String toString() {
            return "Data [inputs=" + Arrays.toString(inputs) + ", label=" + label + ", output=" + output + ", predict="
                    + predict + ", significance=" + significance + ", subsampleWeights="
                    + Arrays.toString(subsampleWeights) + "]";
        }

    }

}
