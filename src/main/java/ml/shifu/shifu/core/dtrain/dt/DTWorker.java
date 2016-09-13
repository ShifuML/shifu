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
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ml.shifu.guagua.ComputableMonitor;
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
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams.NodeStats;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.distribution.PoissonDistribution;
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
 * into memory. To store predict and label in GBDT, In {@link Data} predict and label are all set even with RF. Data are
 * stored as float types to save memory consumption.
 * 
 * <p>
 * For GBDT, when a new tree is transferred to worker, data predict and label are all updated and such value can be
 * covered according to trees and learning rate.
 * 
 * <p>
 * For RF, bagging with replacement are enabled by {@link PoissonDistribution} fields {@link #rng}.
 * 
 * <p>
 * Weighted training are supported in our RF and GBDT impl, in such worker, data.significance is weight set from input.
 * If no weight, such value is set to 1.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 240)
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
     * Basic numeric input count
     */
    protected int numericInputCount;

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
     * Basic output node count for NN model
     */
    protected int outputNodeCount;

    /**
     * {@link #candidateCount} is used to check if no variable is selected. If {@link #inputNodeCount} equals
     * {@link #candidateCount}, which means no column is selected or all columns are selected.
     */
    protected int candidateCount;

    /**
     * input record size, inc one by one.
     */
    protected long count;

    /**
     * sampled input record size.
     */
    protected long sampleCount;

    /**
     * Training data set with only in memory because for GBDT data will be changed in later iterations.
     */
    private volatile MemoryLimitedList<Data> trainingData;

    /**
     * Validation data set with only in memory because for GBDT data will be changed in later iterations.
     */
    private volatile MemoryLimitedList<Data> validationData;

    /**
     * PoissonDistribution which is used for poission sampling for bagging with replacement.
     */
    protected PoissonDistribution[] rng = null;

    /**
     * If tree number = 1, no need bagging with replacement.
     */
    private Random random = new Random();

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    protected static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR)
            .trimResults();

    /**
     * Index map in which column index and numeric input array index for fast location.
     */
    private Map<Integer, Integer> numericInputIndexMap = new HashMap<Integer, Integer>();

    /**
     * Index map in which column index and categorical input array index for fast location.
     */
    private Map<Integer, Integer> categoricalInputIndexMap = new HashMap<Integer, Integer>();

    /**
     * A map with internal BinCategory map which stores index per each category.
     */
    private Map<Integer, Map<String, Integer>> categoryIndexMap = new HashMap<Integer, Map<String, Integer>>();

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
    private boolean gbdtSampleWithReplacement = true;

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
     * Indicates if there are cross validation data sets.
     */
    protected boolean isCrossValidation = false;


    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled;

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
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

        for(ColumnConfig config: this.columnConfigList) {
            if(config.isCategorical()) {
                Map<String, Integer> categoryMap = new HashMap<String, Integer>();
                for(int i = 0; i < config.getBinCategory().size(); i++) {
                    categoryMap.put(config.getBinCategory().get(i), i);
                }
                this.categoryIndexMap.put(config.getColumnNum(), categoryMap);
            }
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

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams());
        Map<String, Object> validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(this.trainerId);
            LOG.info("Start grid search worker with params: {}", validParams);
        }

        this.treeNum = Integer.valueOf(validParams.get("TreeNum").toString());

        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);

        double validationRate = this.modelConfig.getCrossValidationRate();
        if(Double.compare(validationRate, 0d) != 0) {
            this.trainingData = new MemoryLimitedList<Data>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * (1 - validationRate)),
                    new LinkedList<Data>());
            this.validationData = new MemoryLimitedList<Data>(
                    (long) (Runtime.getRuntime().maxMemory() * memoryFraction * validationRate), new LinkedList<Data>());
        } else {
            this.trainingData = new MemoryLimitedList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                    new LinkedList<Data>());
        }
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.numericInputCount = inputOutputIndex[0];
        this.categoricalInputCount = inputOutputIndex[1];
        this.outputNodeCount = modelConfig.isRegression() ? inputOutputIndex[2] : modelConfig.getTags().size();
        this.isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        this.isCrossValidation = StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath());

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

        // TODO, using reflection
        String lossStr = validParams.get("Loss").toString();
        if(lossStr.equalsIgnoreCase("log")) {
            this.loss = new LogLoss();
        } else if(lossStr.equalsIgnoreCase("absolute")) {
            this.loss = new AbsoluteLoss();
        } else if(lossStr.equalsIgnoreCase("halfgradsquared")) {
            this.loss = new HalfGradSquaredLoss();
        } else {
            this.loss = new SquaredLoss();
        }

        if(this.isGBDT) {
            this.learningRate = Double.valueOf(validParams.get(NNTrainer.LEARNING_RATE).toString());
            Object swrObj = validParams.get("SampleWithReplacement");
            if(swrObj != null) {
                this.gbdtSampleWithReplacement = Boolean.TRUE.toString().equalsIgnoreCase(swrObj.toString());
            }
        }

        if(this.isRF || (this.isGBDT && this.gbdtSampleWithReplacement)) {
            this.rng = new PoissonDistribution[treeNum];
            for(int i = 0; i < treeNum; i++) {
                this.rng[i] = new PoissonDistribution(this.modelConfig.getTrain().getBaggingSampleRate());
            }
        }

        LOG.info(
                "Worker init params:isAfterVarSel={}, treeNum={}, impurity={}, loss={}, learningRate={}, gbdtSampleWithReplacement={}, isRF={}, isGBDT={}",
                isAfterVarSelect, treeNum, impurity.getClass().getName(), loss.getClass().getName(), this.learningRate,
                this.gbdtSampleWithReplacement, this.isRF, this.isGBDT);
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
            this.random = new Random();
        }

        for(Data data: this.trainingData) {
            if(this.isRF) {
                for(TreeNode treeNode: trees) {
                    Node predictNode = predictNodeIndex(treeNode.getNode(), data);
                    if(predictNode.getPredict() != null) {
                        // only update when not in first node, for treeNode, no predict statistics at that time
                        trainError += data.significance
                                * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                        weightedTrainCount += data.significance;
                    }
                }
            }

            if(this.isGBDT) {
                if(this.isContinuousEnabled && lastMasterResult.isContinuousRunningStart()) {
                    recoverGBTData(context, data.output, data.predict, data);
                    trainError += data.significance * loss.computeError(data.predict, data.label);
                    weightedTrainCount += data.significance;
                } else {
                    int currTreeIndex = trees.size() - 1;
                    if(lastMasterResult.isSwitchToNextTree()) {
                        if(currTreeIndex >= 1) {
                            Node node = trees.get(currTreeIndex - 1).getNode();
                            Node predictNode = predictNodeIndex(node, data);
                            if(predictNode.getPredict() != null) {
                                double predict = predictNode.getPredict().getPredict();
                                if(currTreeIndex == 1) {
                                    data.predict = (float) predict;
                                } else {
                                    data.predict += (float) (this.learningRate * predict);
                                }
                                data.output = -1f * loss.computeGradient(data.predict, data.label);
                            }
                            if(!this.gbdtSampleWithReplacement) {
                                // renew next subsample rate
                                if(random.nextDouble() <= modelConfig.getTrain().getBaggingSampleRate()) {
                                    data.subsampleWeights[currTreeIndex % data.subsampleWeights.length] = 1f;
                                } else {
                                    data.subsampleWeights[currTreeIndex % data.subsampleWeights.length] = 0f;
                                }
                            }
                        }
                    }
                    Node predictNode = predictNodeIndex(trees.get(currTreeIndex).getNode(), data);
                    if(currTreeIndex >= 1) {
                        trainError += data.significance * loss.computeError(data.predict, data.label);
                        weightedTrainCount += data.significance;
                    } else {
                        if(predictNode.getPredict() != null) {
                            trainError += data.significance
                                    * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                            weightedTrainCount += data.significance;
                        }
                    }
                }
            }
        }
        if(validationData != null) {
            for(Data data: this.validationData) {
                if(this.isRF) {
                    for(TreeNode treeNode: trees) {
                        Node predictNode = predictNodeIndex(treeNode.getNode(), data);
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
                        recoverGBTData(context, data.output, data.predict, data);
                        validationError += data.significance * loss.computeError(data.predict, data.label);
                        weightedValidationCount += data.significance;
                    } else {
                        int currTreeIndex = trees.size() - 1;
                        if(lastMasterResult.isSwitchToNextTree()) {
                            if(currTreeIndex >= 1) {
                                Node node = trees.get(currTreeIndex - 1).getNode();
                                Node predictNode = predictNodeIndex(node, data);
                                if(predictNode.getPredict() != null) {
                                    double predict = predictNode.getPredict().getPredict();
                                    if(currTreeIndex == 1) {
                                        data.predict = (float) predict;
                                    } else {
                                        data.predict += (float) (this.learningRate * predict);
                                    }
                                    data.output = -1f * loss.computeGradient(data.predict, data.label);
                                }
                                if(!this.gbdtSampleWithReplacement) {
                                    // renew next subsample rate
                                    if(random.nextDouble() <= modelConfig.getTrain().getBaggingSampleRate()) {
                                        data.subsampleWeights[currTreeIndex % data.subsampleWeights.length] = 1f;
                                    } else {
                                        data.subsampleWeights[currTreeIndex % data.subsampleWeights.length] = 0f;
                                    }
                                }
                            }
                        }
                        Node predictNode = predictNodeIndex(trees.get(currTreeIndex).getNode(), data);
                        if(currTreeIndex >= 1) {
                            validationError += data.significance * loss.computeError(data.predict, data.label);
                            weightedValidationCount += data.significance;
                        } else {
                            if(predictNode.getPredict() != null) {
                                validationError += data.significance
                                        * loss.computeError((float) (predictNode.getPredict().getPredict()), data.label);
                                weightedValidationCount += data.significance;
                            }
                        }
                    }
                }
            }
        }

        CompletionService<Map<Integer, NodeStats>> completionService = new ExecutorCompletionService<Map<Integer, NodeStats>>(
                this.threadPool);

        Set<Entry<Integer, TreeNode>> treeNodeEntrySet = todoNodes.entrySet();
        Iterator<Entry<Integer, TreeNode>> treeNodeIterator = treeNodeEntrySet.iterator();
        int roundNodeNumer = treeNodeEntrySet.size() / this.workerThreadCount;
        int modeNodeNumber = treeNodeEntrySet.size() % this.workerThreadCount;
        int realThreadCount = 0;
        LOG.info("while todo size {}", todoNodes.size());
        for(int i = 0; i < this.workerThreadCount; i++) {
            final Map<Integer, TreeNode> localTodoNodes = new HashMap<Integer, TreeNode>();
            final Map<Integer, NodeStats> localStatistics = new HashMap<Integer, DTWorkerParams.NodeStats>();
            for(int j = 0; j < roundNodeNumer; j++) {
                Entry<Integer, TreeNode> tmpTreeNode = treeNodeIterator.next();
                localTodoNodes.put(tmpTreeNode.getKey(), tmpTreeNode.getValue());
                localStatistics.put(tmpTreeNode.getKey(), statistics.get(tmpTreeNode.getKey()));
            }
            if(modeNodeNumber > 0) {
                Entry<Integer, TreeNode> tmpTreeNode = treeNodeIterator.next();
                localTodoNodes.put(tmpTreeNode.getKey(), tmpTreeNode.getValue());
                localStatistics.put(tmpTreeNode.getKey(), statistics.get(tmpTreeNode.getKey()));
                modeNodeNumber -= 1;
            }
            LOG.info("thread {} todo size {} stats size {} ", i, localTodoNodes.size(), localStatistics.size());

            if(localTodoNodes.size() == 0) {
                continue;
            }
            realThreadCount += 1;
            completionService.submit(new Callable<Map<Integer, NodeStats>>() {
                @Override
                public Map<Integer, NodeStats> call() throws Exception {
                    List<Integer> nodeIndexes = new ArrayList<Integer>(trees.size());
                    for(Data data: DTWorker.this.trainingData) {
                        nodeIndexes.clear();
                        if(DTWorker.this.isRF) {
                            for(TreeNode treeNode: trees) {
                                Node predictNode = predictNodeIndex(treeNode.getNode(), data);
                                nodeIndexes.add(predictNode.getId());
                            }
                        }

                        if(DTWorker.this.isGBDT) {
                            int currTreeIndex = trees.size() - 1;
                            Node predictNode = predictNodeIndex(trees.get(currTreeIndex).getNode(), data);
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
                                    ColumnConfig config = DTWorker.this.columnConfigList.get(columnNum);
                                    double[] featuerStatistic = localStatistics.get(entry.getKey())
                                            .getFeatureStatistics().get(columnNum);
                                    float weight = data.subsampleWeights[treeId % data.subsampleWeights.length];
                                    if(config.isNumerical()) {
                                        float value = data.numericInputs[DTWorker.this.numericInputIndexMap
                                                .get(columnNum)];
                                        int binIndex = getBinIndex(value, config.getBinBoundary());
                                        DTWorker.this.impurity.featureUpdate(featuerStatistic, binIndex, data.output,
                                                data.significance, weight);
                                    } else if(config.isCategorical()) {
                                        String category = data.categoricalInputs[DTWorker.this.categoricalInputIndexMap
                                                .get(columnNum)];
                                        Integer binIndex = DTWorker.this.categoryIndexMap.get(columnNum).get(category);
                                        if(binIndex == null) {
                                            // add to null bin which is the last one
                                            binIndex = config.getBinCategory().size();
                                        }
                                        DTWorker.this.impurity.featureUpdate(featuerStatistic, binIndex, data.output,
                                                data.significance, weight);
                                    } else {
                                        throw new IllegalStateException(
                                                "Only numerical and categorical columns supported. ");
                                    }
                                }
                            }
                        }
                    }
                    return localStatistics;
                }
            });
        }

        int rCnt = 0;
        while(rCnt < realThreadCount) {
            try {
                statistics.putAll(completionService.take().get());
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            rCnt += 1;
        }

        LOG.info(
                "worker count is {}, error is {}, and stats size is {}. weightedTrainCount {}, weightedValidationCount {}, trainError {}, validationError {}",
                count, trainError, statistics.size(), weightedTrainCount, weightedValidationCount, trainError,
                validationError);
        return new DTWorkerParams(weightedTrainCount, weightedValidationCount, trainError, validationError, statistics);
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
        LOG.info("    - # Records of the Master Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        LOG.info("        - Cross Validation Rate: {}.", this.modelConfig.getCrossValidationRate());
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.size());
        if(validationData != null) {
            LOG.info("        - # Records of the Validation Set: {}.", this.validationData.size());
        }
    }

    private List<Integer> getAllValidFeatures() {
        List<Integer> features = new ArrayList<Integer>();
        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    features.add(config.getColumnNum());
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                    features.add(config.getColumnNum());
                }
            }
        }
        return features;
    }

    /**
     * 'binBoundary' is ArrayList in fact, so we can use get method. ["-Infinity", 1d, 4d, ....]
     */
    public static int getBinIndex(float value, List<Double> binBoundary) {
        if(binBoundary.size() <= 1) {
            throw new IllegalArgumentException();
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

    private Node predictNodeIndex(Node node, Data data) {
        Node currNode = node;
        Split split = currNode.getSplit();
        if(split == null || currNode.isLeaf()) {
            return currNode;
        }

        ColumnConfig columnConfig = this.columnConfigList.get(split.getColumnNum());

        Node nextNode = null;
        if(columnConfig.isNumerical()) {
            float value = data.numericInputs[this.numericInputIndexMap.get(split.getColumnNum())];
            if(value <= split.getThreshold()) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        } else if(columnConfig.isCategorical()) {
            String value = data.categoricalInputs[this.categoricalInputIndexMap.get(split.getColumnNum())];
            if(split.getLeftCategories().contains(value)) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        }

        assert nextNode != null;
        return predictNodeIndex(nextNode, data);
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<DTMasterParams, DTWorkerParams> context) {
        this.count += 1;
        if((this.count) % 5000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        double baggingSampleRate = this.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hashcode after parsing.
        if(!modelConfig.isFixInitialInput() && Double.compare(Math.random(), baggingSampleRate) >= 0) {
            return;
        }

        // hashcode for fixed input split in train and validation
        long hashcode = 0;

        float[] numericInputs = new float[this.numericInputCount];
        String[] categoricalInputs = new String[this.categoricalInputCount];
        float ideal = 0f;
        float significance = 1f;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, numericInputsIndex = 0, categoricalInputsIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
            float floatValue = input.length() == 0 ? 0f : NumberFormatUtils.getFloat(input, 0f);
            // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
            floatValue = (Float.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;
            if(index == this.columnConfigList.size()) {
                // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 1f)
                significance = input.length() == 0 ? 1f : NumberFormatUtils.getFloat(input, 1f);
                // the last field is significance, break here
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    ideal = floatValue;
                } else {
                    if(!isAfterVarSelect) {
                        // no variable selected, good candidate but not meta and not target chose
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            if(columnConfig.isNumerical()) {
                                numericInputs[numericInputsIndex] = floatValue;
                                this.numericInputIndexMap.put(columnConfig.getColumnNum(), numericInputsIndex);
                                hashcode = hashcode * 31 + Double.valueOf(floatValue).hashCode();
                                numericInputsIndex += 1;
                            } else if(columnConfig.isCategorical()) {
                                if(input == null || input.length() == 0) {
                                    // use empty to replace null categories
                                    categoricalInputs[categoricalInputsIndex] = "";
                                } else {
                                    categoricalInputs[categoricalInputsIndex] = input;
                                }
                                this.categoricalInputIndexMap.put(columnConfig.getColumnNum(), categoricalInputsIndex);
                                hashcode = hashcode * 31 + categoricalInputs[categoricalInputsIndex].hashCode();
                                categoricalInputsIndex += 1;
                            }
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            if(columnConfig.isNumerical()) {
                                numericInputs[numericInputsIndex] = floatValue;
                                this.numericInputIndexMap.put(columnConfig.getColumnNum(), numericInputsIndex);
                                hashcode = hashcode * 31 + Double.valueOf(floatValue).hashCode();
                                numericInputsIndex += 1;
                            } else if(columnConfig.isCategorical()) {
                                if(input == null || input.length() == 0) {
                                    // use empty to replace null categories
                                    categoricalInputs[categoricalInputsIndex] = "";
                                } else {
                                    categoricalInputs[categoricalInputsIndex] = input;
                                }
                                this.categoricalInputIndexMap.put(columnConfig.getColumnNum(), categoricalInputsIndex);
                                hashcode = hashcode * 31 + categoricalInputs[categoricalInputsIndex].hashCode();
                                categoricalInputsIndex += 1;
                            }
                        }
                    }
                }
            }
            index += 1;
        }

        // if fixInitialInput = true, we should use hashcode to sample.
        long longBaggingSampleRate = Double.valueOf(baggingSampleRate * 100).longValue();
        if(this.modelConfig.isFixInitialInput() && hashcode % 100 >= longBaggingSampleRate) {
            return;
        }
        this.sampleCount += 1;

        if(this.isOneVsAll) {
            // if one vs all, update target value according to index of target
            ideal = updateOneVsAllTargetValue(ideal);
        }

        float output = ideal;
        float predict = ideal;

        Data data = new Data(numericInputs, categoricalInputs, predict, output, output, significance, sampleWeights());

        boolean isNeedFailOver = !context.isFirstIteration();
        // recover for gbdt fail over
        if(isNeedFailOver && this.isGBDT) {
            recoverGBTData(context, output, predict, data);
        }
        boolean isTesting = false;
        if(context.getAttachment() != null && context.getAttachment() instanceof Boolean) {
            isTesting = (Boolean) context.getAttachment();
        }
        this.addDataPairToDataSet(hashcode,data,isTesting);
    }
    
    protected void addDataPairToDataSet(long hashcode, Data data, boolean isTesting) {
        if(isTesting) {
            this.validationData.append(data);
            return;
        } else if(this.isCrossValidation && (!isTesting)) {
            this.trainingData.append(data);
            return;
        }
        double validationRate = this.modelConfig.getCrossValidationRate();
        if(Double.compare(validationRate, 0d) != 0) {
            if(this.modelConfig.isFixInitialInput()) {
                long longValidation = Double.valueOf(validationRate * 100).longValue();
                if(hashcode % 100 >= longValidation) {
                    this.trainingData.append(data);
                } else {
                    this.validationData.append(data);
                }
            } else {
                if(random.nextDouble() >= validationRate) {
                    this.trainingData.append(data);
                } else {
                    this.validationData.append(data);
                }
            }
        } else {
            this.trainingData.append(data);
        }
    }

    private void recoverGBTData(WorkerContext<DTMasterParams, DTWorkerParams> context, float output, float predict,
            Data data) {
        DTMasterParams lastMasterResult = context.getLastMasterResult();
        if(lastMasterResult != null) {
            List<TreeNode> trees = lastMasterResult.getTrees();
            if(trees.size() > 1) {
                // if isSwitchToNextTree == false, iterate all trees except current one to get new predict and
                // output value; if isSwitchToNextTree == true, iterate all trees except current two trees.
                // the last tree is a root node, the tree with index size-2 will be called in doCompute method
                int iterLen = lastMasterResult.isSwitchToNextTree() ? trees.size() - 2 : trees.size() - 1;
                for(int i = 0; i < iterLen; i++) {
                    TreeNode currTree = trees.get(i);
                    if(i == 0) {
                        double oldPredict = predictNodeIndex(currTree.getNode(), data).getPredict().getPredict();
                        predict = (float) oldPredict;
                        output = -1f * loss.computeGradient(predict, data.label);
                    } else {
                        double oldPredict = predictNodeIndex(currTree.getNode(), data).getPredict().getPredict();
                        predict += (float) (this.learningRate * oldPredict);
                        output = -1f * loss.computeGradient(predict, data.label);
                    }
                }
                data.output = output;
                data.predict = predict;
            }
        }
    }

    private float[] sampleWeights() {
        float[] sampleWeights;
        if(this.treeNum == 1 || (this.isGBDT && !this.gbdtSampleWithReplacement)) {
            // if tree == 1 or GBDT, don't use with replacement sampling; for GBDT, every time is one tree
            sampleWeights = new float[1];
            if(random.nextDouble() <= modelConfig.getTrain().getBaggingSampleRate()) {
                sampleWeights[0] = 1f;
            } else {
                sampleWeights[0] = 0f;
            }
        } else {
            // if gbdt and gbdtSampleWithReplacement = true, still sampling with replacement
            sampleWeights = new float[this.treeNum];
            for(int i = 0; i < sampleWeights.length; i++) {
                sampleWeights[i] = this.rng[i].sample();
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

        float[] numericInputs;
        String[] categoricalInputs;
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
        }

        public Data(float[] numericInputs, String[] categoricalInputs, float predict, float output, float label,
                float significance, float[] subsampleWeights) {
            this.numericInputs = numericInputs;
            this.categoricalInputs = categoricalInputs;
            this.predict = predict;
            this.output = output;
            this.label = label;
            this.significance = significance;
            this.subsampleWeights = subsampleWeights;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(numericInputs.length);
            for(float input: numericInputs) {
                out.writeFloat(input);
            }

            out.writeInt(categoricalInputs.length);
            for(String input: categoricalInputs) {
                out.writeUTF(input);
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
            this.numericInputs = new float[iLen];
            for(int i = 0; i < iLen; i++) {
                this.numericInputs[i] = in.readFloat();
            }

            int cLen = in.readInt();
            this.categoricalInputs = new String[cLen];
            for(int i = 0; i < cLen; i++) {
                this.categoricalInputs[i] = in.readUTF();
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
            return "Data [numericInputs=" + Arrays.toString(numericInputs) + ", categoricalInputs="
                    + Arrays.toString(categoricalInputs) + ", label=" + label + ", output=" + output + ", predict="
                    + predict + ", significance=" + significance + ", subsampleWeights="
                    + Arrays.toString(subsampleWeights) + "]";
        }

    }

}
