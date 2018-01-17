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

import java.io.IOException;
import java.util.*;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConvergeJudger;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.Weight;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.fs.Path;
import org.encog.ml.BasicML;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
     * Convergence threshold setting.
     */
    private double convergenceThreshold = Double.MIN_VALUE;

    /**
     * Convergence judger instance for convergence checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();

    /**
     * Valid params specially for grid search
     */
    private Map<String, Object> validParams;

    /**
     * Validation tolerance which is for early stop, by default it is 0d which means early stop is not enabled.
     */
    private double validationTolerance = 0d;

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
    private boolean isAfterVarSelect;

    /**
     * Weight initializer, can be 'default', 'gaussian' or 'xavier'.
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
        int totalWorkerCount = 0;
        for(NNParams nn: context.getWorkerResults()) {
            totalTestError += nn.getTestError();
            totalTrainError += nn.getTrainError();
            this.globalNNParams.accumulateGradients(nn.getGradients());
            this.globalNNParams.accumulateTrainSize(nn.getTrainSize());
            totalCount += nn.getCount();
            // original worker count before combinable
            totalWorkerCount += nn.getWrCount();
            size++;
        }

        LOG.debug("ELM gradients debug for 0 gradient {}", this.globalNNParams.getGradients()[0]);
        LOG.debug("Total Count is {}. totalWorkerCount is {}", totalCount, totalWorkerCount);

        // worker result size is 0. throw exception because shouldn't happen
        if(size == 0) {
            throw new IllegalArgumentException("workers' results are empty.");
        }

        // initialize weightCalCulater.
        if(this.weightCalculator == null) {
            this.learningRate = this.rawLearningRate;
            this.weightCalculator = new Weight(this.globalNNParams.getGradients().length,
                    this.globalNNParams.getTrainSize(), learningRate, propagation, this.regularizedConstant,
                    RegulationLevel.to(this.validParams.get(CommonConstants.REG_LEVEL_KEY)), this.dropoutRate,
                    this.propagation, this.momentum, this.learningDecay, this.adamBeta1, this.adamBeta2);
        } else {
            this.learningRate = this.learningRate * (1.0d - this.learningDecay);
            // without learningDecay Parameter using sqrt(iteration number) to decrease learning rate
            // this.learningRate = this.learningRate / Math.sqrt(context.getCurrentIteration() -1);
            this.weightCalculator.setLearningRate(this.learningRate);
            this.weightCalculator.setNumTrainSize(this.globalNNParams.getTrainSize());
        }

        double[] oldWeights = Arrays.copyOf(this.globalNNParams.getWeights(), this.globalNNParams.getWeights().length);

        // use last weights and current gradients to calculate, current iteration - 1 to remove 1st iteration for worker
        // data reading
        double[] weights = this.weightCalculator.calculateWeights(this.globalNNParams.getWeights(),
                this.globalNNParams.getGradients(), (context.getCurrentIteration() - 1));

        this.globalNNParams.setWeights(weights);

        // average error
        double currentTestError = totalTestError / totalWorkerCount;
        double currentTrainError = totalTrainError / totalWorkerCount;

        boolean vtTriggered = false;
        // if validationTolerance == 0d, means vt check is not enabled
        if(validationTolerance > 0d) {
            double weightSumSquare = 0d;
            double diffWeightSumSquare = 0d;
            for(int i = 0; i < weights.length; i++) {
                weightSumSquare += Math.pow(weights[i], 2);
                diffWeightSumSquare += Math.pow(weights[i] - oldWeights[i], 2);
            }
            if(Math.pow(diffWeightSumSquare, 0.5) < this.validationTolerance
                    * Math.max(Math.pow(weightSumSquare, 0.5), 1d)) {
                LOG.info("Debug: diffWeightSumSquare {}, weightSumSquare {}, validationTolerance {}",
                        Math.pow(diffWeightSumSquare, 0.5), Math.pow(weightSumSquare, 0.5), validationTolerance);
                vtTriggered = true;
            }
        }

        if(currentTestError < this.bestValidationError) {
            this.bestValidationError = currentTestError;
        }

        LOG.info("NNMaster compute iteration {} ( avg train error {}, avg validation error {} )", new Object[] {
                context.getCurrentIteration(), currentTrainError, currentTestError });

        NNParams params = new NNParams();
        params.setTrainError(currentTrainError);
        params.setTestError(currentTestError);
        // prevent null point
        params.setGradients(new double[0]);
        params.setWeights(weights);
        LOG.debug("master result {} in iteration {}", params, context.getCurrentIteration());

        // Convergence judging part
        double avgErr = (currentTrainError + currentTestError) / 2;

        LOG.info("NNMaster compute iteration {} average error: {}, threshold: {}", context.getCurrentIteration(),
                avgErr, convergenceThreshold);

        if(judger.judge(avgErr, convergenceThreshold) || vtTriggered) {
            LOG.info("NNMaster compute iteration {} converged !", context.getCurrentIteration());
            params.setHalt(true);
        } else {
            LOG.debug("NNMaster compute iteration {} not converged yet !", context.getCurrentIteration());
        }

        return params;
    }

    private NNParams initOrRecoverParams(MasterContext<NNParams, NNParams> context) {
        // read existing model weights
        NNParams params = null;
        try {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            BasicML basicML = CommonUtils.loadModel(modelConfig, modelPath,
                    ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource()));
            BasicFloatNetwork existingModel = (BasicFloatNetwork) CommonUtils.getBasicNetwork(basicML);
            if(existingModel == null) {
                params = initWeights();
                LOG.info("Starting to train model from scratch.");
            } else {
                params = initModelParams(existingModel);
                LOG.info("Starting to train model from existing model {}.", modelPath);
            }
        } catch (IOException e) {
            throw new GuaguaRuntimeException(e);
        }
        return params;
    }

    private NNParams initModelParams(BasicNetwork loadModel) {
        NNParams params = new NNParams();
        params.setTrainError(0);
        params.setTestError(0);
        // prevent null point
        params.setGradients(new double[0]);
        params.setWeights(loadModel.getFlat().getWeights());
        return params;
    }

    @SuppressWarnings({ "unchecked" })
    private NNParams initWeights() {
        NNParams params = new NNParams();

        int[] inputAndOutput = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                this.columnConfigList);
        int featureInputsCnt = DTrainUtils.getFeatureInputsCnt(modelConfig, this.columnConfigList,
                new HashSet<Integer>(this.subFeatures));
        @SuppressWarnings("unused")
        int inputNodeCount = inputAndOutput[0] == 0 ? inputAndOutput[2] : inputAndOutput[0];
        // if is one vs all classification, outputNodeCount is set to 1
        int outputNodeCount = modelConfig.isRegression() ? inputAndOutput[1]
                : (modelConfig.getTrain().isOneVsAll() ? inputAndOutput[1] : modelConfig.getTags().size());
        int numLayers = (Integer) validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) validParams.get(CommonConstants.NUM_HIDDEN_NODES);

        BasicNetwork network = DTrainUtils.generateNetwork(featureInputsCnt, outputNodeCount, numLayers, actFunc,
                hiddenNodeList, true, this.dropoutRate, this.wgtInit);

        params.setTrainError(0);
        params.setTestError(0);
        // prevent null point
        params.setGradients(new double[0]);
        params.setWeights(network.getFlat().getWeights());
        return params;
    }

    @Override
    public void init(MasterContext<NNParams, NNParams> context) {
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

        int trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain()
                .getGridConfigFileContent());
        validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }

        Object vtObj = validParams.get("ValidationTolerance");
        if(vtObj != null) {
            try {
                validationTolerance = Double.parseDouble(vtObj.toString());
                LOG.warn("Validation by tolerance is enabled with value {}.", validationTolerance);
            } catch (NumberFormatException ee) {
                validationTolerance = 0d;
                LOG.warn(
                        "Validation by tolerance isn't enabled because of non numerical value of ValidationTolerance: {}.",
                        vtObj);
            }
        } else {
            LOG.info("Validation by tolerance isn't enabled.");
        }

        Object pObject = validParams.get(CommonConstants.PROPAGATION);
        this.propagation = pObject == null ? "Q" : (String) pObject;
        this.rawLearningRate = Double.valueOf(validParams.get(CommonConstants.LEARNING_RATE).toString());
        Object dropoutRateObj = validParams.get(CommonConstants.DROPOUT_RATE);
        if(dropoutRateObj != null) {
            this.dropoutRate = Double.valueOf(dropoutRateObj.toString());
        }
        LOG.info("'dropoutRate' in master is : {}", this.dropoutRate);

        Object learningDecayO = validParams.get("LearningDecay");
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

        Double threshold = this.modelConfig.getTrain().getConvergenceThreshold();
        this.convergenceThreshold = threshold == null ? Double.MIN_VALUE : threshold.doubleValue();
        LOG.info("Convergence threshold in master is :{}", this.convergenceThreshold);

        this.wgtInit = "default";
        Object wgtInitObj = validParams.get("WeightInitializer");
        if(wgtInitObj != null) {
            this.wgtInit = wgtInitObj.toString();
        }

        this.isContinuousEnabled = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));
        Object rconstant = validParams.get(CommonConstants.LR_REGULARIZED_CONSTANT);
        this.regularizedConstant = NumberFormatUtils.getDouble(rconstant == null ? "" : rconstant.toString(), 0d);

        // check if variables are set final selected
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.isAfterVarSelect = (inputOutputIndex[3] == 1);
        // cache all feature list for sampling features
        this.allFeatures = CommonUtils.getAllFeatureList(columnConfigList, isAfterVarSelect);
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

}
