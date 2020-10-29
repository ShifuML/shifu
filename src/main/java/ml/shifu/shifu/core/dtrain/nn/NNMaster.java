/**
 * Copyright [2012-2014] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.fs.Path;
import org.encog.ml.BasicML;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.Weight;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatFlatNetwork;
import ml.shifu.shifu.core.dtrain.earlystop.AbstractEarlyStopStrategy;
import ml.shifu.shifu.core.dtrain.earlystop.ConvergeAndValidToleranceEarlyStop;
import ml.shifu.shifu.core.dtrain.earlystop.WindowEarlyStop;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.udf.norm.PrecisionType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;

/**
 * {@link NNMaster} is used to accumulate all workers NN parameters.
 *
 * <p>
 * We accumulate all gradients from workers to calculate model weights. And set weights to workers. Then workers use
 * weights to set their models and train for another iteration.
 *
 * <p>
 * This logic follows Encog multi-core implementation.
 *
 * <p>
 * Make sure workers and master use the same initialization weights.
 */
public class NNMaster extends AbstractMasterComputable<NNParams, NNParams> {

    private static final Logger LOG = LoggerFactory.getLogger(NNMaster.class);

    /**
     * Global master NN parameters instance which is used to update model weights by using accumulated gradients.
     */
    private NNParams globalNNParams = new NNParams();

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * To calculate weights according to last weights and accumulated gradients
     */
    private Weight weightCalculator = null;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Propagation type for Encog neural network model setting: Q, B, R, C
     */
    private String propagation = "Q";

    /**
     * Raw learning rate set by model configuration.
     */
    private Double rawLearningRate = 0.1d;

    /**
     * Real learning rate used to train nn model
     */
    private Double learningRate = 0.1d;

    /**
     * L1 and L2 regurized constant
     */
    private double regularizedConstant = 0.0d;

    /**
     * Learning decay setting to decrease learning rate iteration by iteration. Common setting value is from 0 to 0.1
     */
    private double learningDecay = 0d;

    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled = false;

    /**
     * Valid params specially for grid search
     */
    private Map<String, Object> validParams;

    /**
     * The best validation error for error computing
     */
    private double bestValidationError = Double.MAX_VALUE;

    /**
     * Dropout rate which is in [0, 1], default it is 0
     */
    private double dropoutRate = 0d;

    /**
     * Cache all features with feature index for searching
     */
    private List<Integer> allFeatures;

    /**
     * Cache subset features with feature index for searching
     */
    private List<Integer> subFeatures;

    /**
     * If variables are selected, if not, select variables with good candidate.
     */
    @SuppressWarnings("unused")
    private boolean isAfterVarSelect;

    /**
     * Weight initializer, can be 'default', 'gaussian' or 'xavier', 'He' or 'Lecun'.
     */
    private String wgtInit;

    /**
     * Momentum factor in Momentum UpdateRule
     */
    private double momentum = 0.5d;

    /**
     * 'beta1' in Adam optimization, only for Adam
     */
    private double adamBeta1 = 0.9d;

    /**
     * 'beta2' in Adam optimization, only for Adam
     */
    private double adamBeta2 = 0.999d;

    /**
     * NN network structure. we use it us pick dropout node index only.
     */
    private FloatFlatNetwork flatNetwork = null;

    /**
     * Fixed Layers id, used for fine tune
     */
    private List<Integer> fixedLayers = new ArrayList<Integer>();

    /**
     * Fixed bias or not, if user want to fix layer
     */
    private boolean fixedBias = true;

    /**
     * The fixed weight index generated by @fixedLayers
     */
    private Set<Integer> fixedWeightIndexSet;

    /**
     * The number of hidden layers for neural network
     */
    private Integer hiddenLayerNum = 0;

    /**
     * The early stop strategy. If it is null, then early stop is disabled
     */
    private AbstractEarlyStopStrategy earlyStopStrategy;

    /**
     * If precision type supported when sending out of gradients to master
     */
    private PrecisionType precisionType;

    @Override
    public NNParams doCompute(MasterContext<NNParams, NNParams> context) {
        if(context.isFirstIteration()) {
            // For first step, we not only initialize whole context but also return weights to master to make sure all
            // workers and master are using the same weights.
            NNParams params = null;
            if(this.isContinuousEnabled) {
                params = initOrRecoverParams(context);
            } else {
                // first iteration is used to set initial weights
                params = initWeights();
                LOG.info("Starting to train model from scratch.");
            }

            // should be set here to make sure master and workers use the same weights
            this.globalNNParams.setWeights(params.getWeights());
            // for continuous model training, here can be optimized by return null and load model weights in worker by
            // reading HDFS.
            return params;
        }

        if(context.getWorkerResults() == null) {
            throw new IllegalArgumentException("workers' results are null.");
        }

        double totalTestError = 0;
        double totalTrainError = 0;
        int size = 0;

        // before accumulate, reset gradients and train size
        this.globalNNParams.reset();

        long totalCount = 0L;
        double totalTrainSum = 0.0d, totalValidationSum = 0.0d;
        int totalWorkerCount = 0;
        for(NNParams nn: context.getWorkerResults()) {
            totalTestError += nn.getValidationError();
            totalTrainError += nn.getTrainError();
            this.globalNNParams.accumulateGradients(nn.getGradients());
            this.globalNNParams.accumulateTrainSize(nn.getTrainSize());
            totalCount += nn.getCount();
            totalTrainSum += nn.getTrainSum();
            totalValidationSum += nn.getValidationSum();
            // original worker count before combinable
            totalWorkerCount += nn.getWrCount();
            size++;
        }
        this.globalNNParams.setTrainSum(totalTrainSum);
        this.globalNNParams.setValidationSum(totalValidationSum);
        this.globalNNParams.setCount(totalCount);

        LOG.debug("ELM gradients debug for 0 gradient {}", this.globalNNParams.getGradients()[0]);
        LOG.info("Total Count is {}. totalWorkerCount is {}", totalCount, totalWorkerCount);
        LOG.info("Total Train Error is {}. totalTrainSum is {}", totalTrainError, totalTrainSum);
        LOG.info("Total Test Error is {}. totalTestSum is {}", totalTestError, totalValidationSum);

        // worker result size is 0. throw exception because shouldn't happen
        if(size == 0) {
            throw new IllegalArgumentException("workers' results are empty.");
        }

        // initialize weightCalculator.
        if(this.weightCalculator == null) {
            this.learningRate = this.rawLearningRate;
            this.weightCalculator = new Weight(this.globalNNParams.getGradients().length,
                    this.globalNNParams.getTrainSum(), learningRate, propagation, this.regularizedConstant,
                    RegulationLevel.to(this.validParams.get(CommonConstants.REG_LEVEL_KEY)), this.propagation,
                    this.momentum, this.learningDecay, this.adamBeta1, this.adamBeta2, this.fixedWeightIndexSet);
        } else {
            this.learningRate = this.learningRate * (1.0d - this.learningDecay);
            // without learningDecay Parameter using sqrt(iteration number) to decrease learning rate
            // this.learningRate = this.learningRate / Math.sqrt(context.getCurrentIteration() -1);
            this.weightCalculator.setLearningRate(this.learningRate);
            this.weightCalculator.setNumTrainSize(this.globalNNParams.getTrainSum());
        }

        double[] oldWeights = Arrays.copyOf(this.globalNNParams.getWeights(), this.globalNNParams.getWeights().length);

        // use last weights and current gradients to calculate, current iteration - 1 to remove 1st iteration for worker
        // data reading
        double[] weights = this.weightCalculator.calculateWeights(this.globalNNParams.getWeights(),
                this.globalNNParams.getGradients(), (context.getCurrentIteration() - 1));
        if(LOG.isDebugEnabled()) {
            logSameWeights(oldWeights, weights);
        }

        this.globalNNParams.setWeights(weights);

        // average error
        double currentTestError = totalTestError / totalValidationSum;
        double currentTrainError = totalTrainError / totalTrainSum;

        if(currentTestError < this.bestValidationError) {
            this.bestValidationError = currentTestError;
        }

        LOG.info("NNMaster compute iteration {} ( avg train error {}, avg validation error {} )",
                new Object[] { context.getCurrentIteration(), currentTrainError, currentTestError });

        NNParams params = new NNParams();
        params.setTrainError(currentTrainError);
        params.setValidationError(currentTestError);
        // prevent null point
        params.setGradients(new double[0]);
        params.setEvaluatedWeights(oldWeights);
        if(this.precisionType == null) {
            params.setWeights(weights);
        } else {
            params.setWeights(castToPrecision(weights));
        }
        if(this.dropoutRate > 0d) {
            params.setDropoutNodes(dropoutNodes());
        }
        LOG.debug("master result {} in iteration {}", params, context.getCurrentIteration());

        if(earlyStopStrategy != null) {
            boolean isToStopEarly = earlyStopStrategy.shouldEarlyStop(context.getCurrentIteration(), weights,
                    currentTrainError, currentTestError);
            if(isToStopEarly) {
                params.setHalt(true);
            }
        }

        return params;
    }

    private double[] castToPrecision(double[] gradients) {
        for(int i = 0; i < gradients.length; i++) {
            gradients[i] = ((Number)this.precisionType.to(gradients[i])).doubleValue();
        }
        return gradients;
    }

    private void logSameWeights(double[] oldWeights, double[] weights) {
        StringBuilder sameWeightIndices = new StringBuilder();
        for(int i = 0; i < weights.length; i++) {
            if(weights[i] == oldWeights[i]) {
                sameWeightIndices.append(i).append(",");
            }
        }
        LOG.info("Same Weight Indices: {}", sameWeightIndices.toString());
    }

    private NNParams initOrRecoverParams(MasterContext<NNParams, NNParams> context) {
        // read existing model weights
        NNParams params = null;
        try {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            BasicML basicML = ModelSpecLoaderUtils.loadModel(modelConfig, modelPath,
                    ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource(), modelPath));

            params = initWeights();

            BasicFloatNetwork existingModel = (BasicFloatNetwork) ModelSpecLoaderUtils.getBasicNetwork(basicML);
            if(existingModel != null) {
                LOG.info("Starting to train model from existing model {}.", modelPath);
                int mspecCompareResult = new NNStructureComparator().compare(this.flatNetwork, existingModel.getFlat());
                if(mspecCompareResult == 0) { // same model structure
                    params.setWeights(existingModel.getFlat().getWeights());
                    this.fixedWeightIndexSet = getFixedWights(fixedLayers);
                } else if(mspecCompareResult == 1) { // new model structure is larger than existing one
                    this.fixedWeightIndexSet = fitExistingModelIn(existingModel.getFlat(), this.flatNetwork,
                            this.fixedLayers, this.fixedBias);
                } else { // new model structure is smaller, couldn't hold existing one
                    throw new GuaguaRuntimeException("Network changed for recover or continuous training. "
                            + "New network couldn't hold existing network!");
                }
            } else {
                LOG.info("Starting to train model from scratch.");
            }
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        }
        return params;
    }

    @SuppressWarnings({ "unchecked" })
    private NNParams initWeights() {
        NNParams params = new NNParams();
        boolean isLinearTarget = CommonUtils.isLinearTarget(modelConfig, columnConfigList);

        int[] inputAndOutput = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                this.columnConfigList);
        int featureInputsCnt = DTrainUtils.getFeatureInputsCnt(modelConfig, this.columnConfigList,
                new HashSet<Integer>(this.subFeatures));
        @SuppressWarnings("unused")
        int inputNodeCount = inputAndOutput[0] == 0 ? inputAndOutput[2] : inputAndOutput[0];
        // if is one vs all classification, outputNodeCount is set to 1, if classes=2, outputNodeCount is also 1
        int classes = modelConfig.getTags().size();
        int outputNodeCount = (isLinearTarget || modelConfig.isRegression()) ? inputAndOutput[1]
                : (modelConfig.getTrain().isOneVsAll() ? inputAndOutput[1] : (classes == 2 ? 1 : classes));
        int numLayers = (Integer) validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) validParams.get(CommonConstants.NUM_HIDDEN_NODES);

        String outputActivationFunc = (String) validParams.get(CommonConstants.OUTPUT_ACTIVATION_FUNC);
        BasicNetwork network = DTrainUtils.generateNetwork(featureInputsCnt, outputNodeCount, numLayers, actFunc,
                hiddenNodeList, true, this.dropoutRate, this.wgtInit,
                CommonUtils.isLinearTarget(modelConfig, columnConfigList), outputActivationFunc);

        this.flatNetwork = (FloatFlatNetwork) network.getFlat();

        params.setTrainError(0);
        params.setValidationError(0);
        // prevent null point
        params.setGradients(new double[0]);
        if(this.precisionType == null) {
            params.setWeights(network.getFlat().getWeights());
        } else {
            params.setWeights(castToPrecision(network.getFlat().getWeights()));
        }
        return params;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public void init(MasterContext<NNParams, NNParams> context) {
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

        // load precision type, if not set, leave it to null
        String precision = props.getProperty(Constants.SHIFU_PRECISION_TYPE);
        if(precision != null) {
            this.precisionType = PrecisionType.of(precision);
        }

        int trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                modelConfig.getTrain().getGridConfigFileContent());
        validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }

        Boolean enabledEarlyStop = DTrainUtils.getBoolean(validParams, CommonConstants.ENABLE_EARLY_STOP,
                Boolean.FALSE);
        if(enabledEarlyStop) {
            Double validTolerance = DTrainUtils.getDouble(validParams, CommonConstants.VALIDATION_TOLERANCE, null);
            if(validTolerance == null) {
                LOG.info("Early Stop is enabled. use WindowEarlyStop");
                // windowSize default 20, user should could adjust it
                this.earlyStopStrategy = new WindowEarlyStop(context, this.modelConfig,
                        DTrainUtils.getInt(context.getProps(), CommonConstants.SHIFU_TRAIN_EARLYSTOP_WINDOW_SIZE, 20));
            } else {
                LOG.info("Early Stop is enabled. use ConvergeAndValiToleranceEarlyStop");
                Double threshold = this.modelConfig.getTrain().getConvergenceThreshold();
                this.earlyStopStrategy = new ConvergeAndValidToleranceEarlyStop(
                        threshold == null ? Double.MIN_VALUE : threshold.doubleValue(), validTolerance);
            }
        }

        Object pObject = validParams.get(CommonConstants.PROPAGATION);
        this.propagation = pObject == null ? "Q" : (String) pObject;
        this.rawLearningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());
        Object dropoutRateObj = validParams.get(CommonConstants.DROPOUT_RATE);
        if(dropoutRateObj != null) {
            this.dropoutRate = Double.valueOf(dropoutRateObj.toString());
        }
        LOG.info("'dropoutRate' in master is : {}", this.dropoutRate);

        Object learningDecayO = validParams.get(CommonConstants.LEARNING_DECAY);
        if(learningDecayO != null) {
            this.learningDecay = Double.valueOf(learningDecayO.toString());
        }
        LOG.info("'learningDecay' in master is :{}", learningDecay);

        Object momentumO = validParams.get("Momentum");
        if(momentumO != null) {
            this.momentum = Double.valueOf(momentumO.toString());
        }
        LOG.info("'momentum' in master is :{}", momentum);

        Object adamBeta1O = validParams.get("AdamBeta1");
        if(adamBeta1O != null) {
            this.adamBeta1 = Double.valueOf(adamBeta1O.toString());
        }
        LOG.info("'adamBeta1' in master is :{}", adamBeta1);

        Object adamBeta2O = validParams.get("AdamBeta2");
        if(adamBeta2O != null) {
            this.adamBeta2 = Double.valueOf(adamBeta2O.toString());
        }
        LOG.info("'adamBeta2' in master is :{}", adamBeta2);

        this.wgtInit = "default";
        Object wgtInitObj = validParams.get(CommonConstants.WEIGHT_INITIALIZER);
        if(wgtInitObj != null) {
            this.wgtInit = wgtInitObj.toString();
        }

        this.isContinuousEnabled = Boolean.TRUE.toString()
                .equalsIgnoreCase(context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));
        Object rconstant = validParams.get(CommonConstants.REGULARIZED_CONSTANT);
        this.regularizedConstant = NumberFormatUtils.getDouble(rconstant == null ? "" : rconstant.toString(), 0d);

        // We do not update weight in fixed layers so that we could fine tune other layers of NN
        Object fixedLayers2O = validParams.get(CommonConstants.FIXED_LAYERS);
        if(fixedLayers2O != null) {
            this.fixedLayers = (List<Integer>) fixedLayers2O;
        }
        LOG.info("Fixed layers in master is :{}", this.fixedLayers.toString());

        Object fixedBiasObj = validParams.getOrDefault(CommonConstants.FIXED_BIAS, true);
        this.fixedBias = (Boolean) fixedBiasObj;

        Object hiddenLayerNumObj = validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        if(hiddenLayerNumObj != null && StringUtils.isNumeric(hiddenLayerNumObj.toString())) {
            this.hiddenLayerNum = Integer.valueOf(hiddenLayerNumObj.toString());
        }
        LOG.info("hiddenLayerNum in master is :{}", this.hiddenLayerNum);

        // check if variables are set final selected
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        // cache all feature list for sampling features
        this.allFeatures = new ArrayList<>(DTrainUtils.generateModelFeatureSet(modelConfig, columnConfigList));
        String subsetStr = context.getProps().getProperty(CommonConstants.SHIFU_NN_FEATURE_SUBSET);
        if(StringUtils.isBlank(subsetStr)) {
            this.subFeatures = this.allFeatures;
        } else {
            String[] splits = subsetStr.split(",");
            this.subFeatures = new ArrayList<Integer>(splits.length);
            for(String split: splits) {
                this.subFeatures.add(Integer.parseInt(split));
            }
        }

        // recover master states here is globalNNParams
        // not init but not first iteration, first recover from last master result set from guagua
        if(!context.isFirstIteration()) {
            NNParams params = context.getMasterResult();
            if(params != null && params.getWeights() != null) {
                this.globalNNParams.setWeights(params.getWeights());
            } else {
                // else read from checkpoint
                params = initOrRecoverParams(context);
                this.globalNNParams.setWeights(params.getWeights());
            }
        }
    }

    private HashSet<Integer> dropoutNodes() {
        Random random = new Random(System.currentTimeMillis());

        HashSet<Integer> droppedNodeIndices = new HashSet<Integer>();

        // from input to last hidden layer. (exclude output layer)
        for(int i = this.flatNetwork.getLayerIndex().length - 1; i > 0; i--) {
            int beginNeuronIndex = this.flatNetwork.getLayerIndex()[i];
            // exclude constant neuron
            int neuronCount = this.flatNetwork.getLayerFeedCounts()[i];

            // from first neuron to last neuron in current layer
            for(int j = 0; j < neuronCount; j++) {
                if(random.nextDouble() < this.flatNetwork.getLayerDropoutRates()[i]) {
                    // drop this node by adding it into list and will passing
                    // this list to workers
                    droppedNodeIndices.add(beginNeuronIndex + j);
                }
            }
        }

        LOG.debug("layerIndex:{}; layerCounts:{}; dropoutNodes:{}", Arrays.toString(this.flatNetwork.getLayerIndex()),
                Arrays.toString(this.flatNetwork.getLayerCounts()),
                Arrays.toString(droppedNodeIndices.toArray(new Integer[droppedNodeIndices.size()])));
        return droppedNodeIndices;
    }

    /**
     * User's input fixed layer ID is different from ours. we need to use hiddenLayerNum to do transformation.
     * For example, when user what to fix first hidden layer, 2 -> his.hiddenLayerNum - 2 + 1
     * <p>
     * fixed layer id represent the weights between neural network layers. See below
     * <p>
     * input hidden output
     * o
     * o o
     * o (layer1) o (layer2) o
     * o o
     * o
     * <p>
     * fixed layer cannot be output layer and input layer, which does not have meanings
     *
     * @param fixedLayers
     * @return
     */
    private Set<Integer> getFixedWights(List<Integer> fixedLayers) {
        Set<Integer> fixedWeight = new HashSet<Integer>();

        for(int fixedLayer: fixedLayers) {
            int realLayer = this.hiddenLayerNum - fixedLayer + 1;
            int fromWeightIndex = this.flatNetwork.getWeightIndex()[realLayer];
            int toWeightIndex = this.flatNetwork.getWeightIndex()[realLayer + 1];
            for(int index = fromWeightIndex; index < toWeightIndex; index++) {
                fixedWeight.add(index);
            }
        }

        return fixedWeight;
    }

    /**
     * Fit one FlatNetwork (fromFlatNetwork) to another FlatNetwork (toFlatNetwork), and if there is fixedLayers
     * return the weight index in the new FlatNetwork
     * <p>
     * Please Note - the destination FlatNetwork should be larger than source FlatNetwork. Or it will generate
     * incorrect weight mapping.
     *
     * @param fromFlatNetwork
     *            - the source FlatNetwork (smaller network)
     * @param toFlatNetwork
     *            - the destination FlatNetwork (larger network)
     * @param fixedLayers
     *            - the fixed lays in source FlatNetwork
     * @param fixedBias
     *            - the bias is fixed, if user want to fix layer
     * @return the fixed weight index in destination FlatNetwork. If there is no fixed layers, it will return
     *         empty set.
     */
    public Set<Integer> fitExistingModelIn(FlatNetwork fromFlatNetwork, FlatNetwork toFlatNetwork,
            List<Integer> fixedLayers, boolean fixedBias) {
        Set<Integer> fixedWeightIndexSet = new HashSet<Integer>();

        for(int layer = fromFlatNetwork.getLayerIndex().length - 1; layer > 0; layer--) {
            int fromLayerOutputCnt = fromFlatNetwork.getLayerFeedCounts()[layer - 1];
            int fromLayerInputCnt = fromFlatNetwork.getLayerCounts()[layer];

            int toLayer = toFlatNetwork.getLayerIndex().length - (fromFlatNetwork.getLayerIndex().length - layer);
            int toLayerInputCnt = toFlatNetwork.getLayerCounts()[toLayer];

            int fromIndexPos = fromFlatNetwork.getWeightIndex()[layer - 1];
            int toIndexPos = toFlatNetwork.getWeightIndex()[toLayer - 1];

            int realLayer = (fromFlatNetwork.getLayerIndex().length - layer);
            boolean isFixedLayer = (CollectionUtils.isNotEmpty(fixedLayers) && fixedLayers.contains(realLayer));

            for(int i = 0; i < fromLayerOutputCnt; i++) {
                for(int j = 0; j < fromLayerInputCnt; j++) {
                    int fromWeightIndex = fromIndexPos + (i * fromLayerInputCnt) + j;
                    int toWeightIndex = toIndexPos + (i * toLayerInputCnt) + j;
                    if(j == fromLayerInputCnt - 1) { // move bias to end
                        toWeightIndex = toIndexPos + (i * toLayerInputCnt) + (toLayerInputCnt - 1);
                    }

                    toFlatNetwork.getWeights()[toWeightIndex] = fromFlatNetwork.getWeights()[fromWeightIndex];

                    if(isFixedLayer) { // is this layer that user want to fix?
                        if(fixedBias || j < fromLayerInputCnt - 1) {
                            // if user want to fix bias also, or the weight is not for bias
                            fixedWeightIndexSet.add(toWeightIndex);
                        }
                    }

                    // System.out.println(fromWeightIndex + " -> " + toWeightIndex);
                }
            }
        }

        return fixedWeightIndexSet;
    }

    public Set<Integer> fitExistingModelIn(FlatNetwork fromFlatNetwork, FlatNetwork toFlatNetwork,
            List<Integer> fixedLayers) {
        return fitExistingModelIn(fromFlatNetwork, toFlatNetwork, fixedLayers, true);
    }
}