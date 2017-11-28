/*
 * Copyright [2013-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.dtrain.lr;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConvergeJudger;
import ml.shifu.shifu.core.LR;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.Weight;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link LogisticRegressionMaster} defines logic to update global <a
 * href=http://en.wikipedia.org/wiki/Logistic_regression >logistic regression</a> model.
 * 
 * <p>
 * At first iteration, master builds a random model then send to all workers to start computing. This is to make all
 * workers use the same model at the starting time.
 * 
 * <p>
 * At other iterations, master works:
 * <ul>
 * <li>1. Accumulate all gradients from workers.</li>
 * <li>2. Update global models by using accumulated gradients.</li>
 * <li>3. Send new global model to workers by returning model parameters.</li>
 * </ul>
 * 
 * <p>
 * L1 and l2 regulations are supported by configuration: RegularizedConstant in model params of ModelConfig.json.
 */
public class LogisticRegressionMaster extends
        AbstractMasterComputable<LogisticRegressionParams, LogisticRegressionParams> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionMaster.class);

    /**
     * Input column number without bias
     */
    private int inputNum;

    /**
     * This is the model weights in LR which will be updated each iteration TODO, if master is failed, how to recovery
     */
    private double[] weights;

    /**
     * Learning rate configured by user in params
     */
    private double learningRate = 1.0d;

    /**
     * Regulation parameter for l1 or l2
     */
    private double regularizedConstant = 0.0d;

    /**
     * To calculate weights according to last weights and accumulated gradients
     */
    private Weight weightCalculator = null;

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Convergence threshold setting by user in ModelConfig.json.
     */
    private double convergenceThreshold;

    /**
     * Convergence judger instance for convergence checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();

    /**
     * Propagation type for lr model setting: Q, B, R, C
     */
    private String propagation = "Q";

    /**
     * Whether some configurations are initialized
     */
    private AtomicBoolean isInitialized = new AtomicBoolean(false);

    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled = false;

    /**
     * Validation tolerance which is for early stop, by default it is 0d which means early stop is not enabled.
     */
    private double validationTolerance = 0d;

    /**
     * The best validation error for error computing
     */
    private double bestValidationError = Double.MAX_VALUE;

    /**
     * Valid params specially for grid search
     */
    private Map<String, Object> validParams;

    @Override
    public void init(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        loadConfigFiles(context.getProps());
        int trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain()
                .getGridConfigFileContent());
        validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }

        this.learningRate = Double.valueOf(this.validParams.get(CommonConstants.LR_LEARNING_RATE).toString());
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(),
                this.columnConfigList);
        this.inputNum = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];

        Object vtObj = this.validParams.get("ValidationTolerance");
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

        Double threshold = this.modelConfig.getTrain().getConvergenceThreshold();
        this.convergenceThreshold = threshold == null ? 0d : threshold.doubleValue();
        LOG.info("Convergence threshold in master is :{}", this.convergenceThreshold);

        Object pObject = validParams.get(CommonConstants.PROPAGATION);
        this.propagation = pObject == null ? "Q" : (String) pObject;

        Object rconstant = validParams.get(CommonConstants.LR_REGULARIZED_CONSTANT);
        this.regularizedConstant = NumberFormatUtils.getDouble(rconstant == null ? "" : rconstant.toString(), 0d);

        this.isContinuousEnabled = Boolean.TRUE.toString().equalsIgnoreCase(
                context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));
        LOG.info("continuousEnabled: {}", this.isContinuousEnabled);

        // not initialized and not first iteration, should be fault tolerence, recover state in LogisticRegressionMaster
        if(!context.isFirstIteration()) {
            LogisticRegressionParams lastMasterResult = context.getMasterResult();
            if(lastMasterResult != null && lastMasterResult.getParameters() != null) {
                // recover state in current master computable and return to workers
                this.weights = lastMasterResult.getParameters();
            } else {
                // no weights, restarted from the very beginning, this may not happen
                this.weights = initWeights().getParameters();
            }
        }
    }

    private LogisticRegressionParams initModelParams(LR loadModel) {
        LogisticRegressionParams params = new LogisticRegressionParams();
        params.setTrainError(0);
        params.setTestError(0);
        // prevent null point
        this.weights = loadModel.getWeights();
        params.setParameters(this.weights);
        return params;
    }

    private LogisticRegressionParams initOrRecoverParams(
            MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        LOG.info("read from existing model");
        LogisticRegressionParams params = null;
        // read existing model weights
        try {
            Path modelPath = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            LR existingModel = (LR) CommonUtils.loadModel(modelConfig, modelPath,
                    ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource()));
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

    @Override
    public LogisticRegressionParams doCompute(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(isInitialized.compareAndSet(false, true)) {
            // not initialized and not first iteration, should be fault tolerance, recover state in
            // LogisticRegressionMaster
            if(!context.isFirstIteration()) {
                LogisticRegressionParams lastMasterResult = context.getMasterResult();
                if(lastMasterResult != null && lastMasterResult.getParameters() != null) {
                    // recover state in current master computable and return to workers
                    this.weights = lastMasterResult.getParameters();
                    return lastMasterResult;
                } else {
                    // no weights, restarted from the very beginning, this may not happen
                    return initWeights();
                }
            }
        }

        if(context.isFirstIteration()) {
            if(this.isContinuousEnabled) {
                return initOrRecoverParams(context);
            } else {
                return initWeights();
            }
        } else {
            // append bias
            double[] gradients = new double[this.inputNum + 1];
            double trainError = 0.0d, testError = 0d;
            long trainSize = 0, testSize = 0;
            for(LogisticRegressionParams param: context.getWorkerResults()) {
                if(param != null) {
                    for(int i = 0; i < gradients.length; i++) {
                        gradients[i] += param.getParameters()[i];
                    }
                    trainError += param.getTrainError();
                    testError += param.getTestError();
                    trainSize += param.getTrainSize();
                    testSize += param.getTestSize();
                }
            }

            if(this.weightCalculator == null) {
                this.weightCalculator = new Weight(weights.length, trainSize, learningRate, this.propagation,
                        this.regularizedConstant, RegulationLevel.to(this.validParams
                                .get(CommonConstants.REG_LEVEL_KEY)), 0d);
            } else {
                this.weightCalculator.setNumTrainSize(trainSize);
            }

            double[] oldWeights = Arrays.copyOf(this.weights, this.weights.length);
            this.weights = this.weightCalculator.calculateWeights(this.weights, gradients,
                    (context.getCurrentIteration() - 1));

            double finalTrainError = trainError / trainSize;
            double finalTestError = testError / testSize;
            LOG.info("Iteration {} with train error {}, test error {}", context.getCurrentIteration(), finalTrainError,
                    finalTestError);
            LogisticRegressionParams lrParams = new LogisticRegressionParams(weights, finalTrainError, finalTestError,
                    trainSize, testSize);

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

            if(finalTestError < this.bestValidationError) {
                this.bestValidationError = finalTestError;
            }

            if(judger.judge(finalTrainError + finalTestError / 2, convergenceThreshold) || vtTriggered) {
                LOG.info("LRMaster compute iteration {} converged !", context.getCurrentIteration());
                lrParams.setHalt(true);
            } else {
                LOG.debug("LRMaster compute iteration {} not converged yet !", context.getCurrentIteration());
            }
            return lrParams;
        }
    }

    private LogisticRegressionParams initWeights() {
        weights = new double[this.inputNum + 1];
        for(int i = 0; i < weights.length; i++) {
            weights[i] = nextDouble(-1, 1);
        }
        return new LogisticRegressionParams(weights);
    }

    private void loadConfigFiles(final Properties props) {
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
    }

    public final double nextDouble(final double min, final double max) {
        final double range = max - min;
        return (range * Math.random()) + min;
    }

}
