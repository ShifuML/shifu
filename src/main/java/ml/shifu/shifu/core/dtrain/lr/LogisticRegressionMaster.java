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

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.LR;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.Weight;
import ml.shifu.shifu.core.dtrain.earlystop.AbstractEarlyStopStrategy;
import ml.shifu.shifu.core.dtrain.earlystop.ConvergeAndValidToleranceEarlyStop;
import ml.shifu.shifu.core.dtrain.earlystop.WindowEarlyStop;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

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
public class LogisticRegressionMaster
        extends AbstractMasterComputable<LogisticRegressionParams, LogisticRegressionParams> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionMaster.class);

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
     * Propagation type for lr model setting: Q, B, R, C
     */
    private String propagation = "R";

    /**
     * Whether some configurations are initialized
     */
    private AtomicBoolean isInitialized = new AtomicBoolean(false);

    /**
     * Whether to enable continuous model training based on existing models.
     */
    private boolean isContinuousEnabled = false;

    /**
     * The best validation error for error computing
     */
    private double bestValidationError = Double.MAX_VALUE;

    /**
     * Valid params specially for grid search
     */
    private Map<String, Object> validParams;

    /**
     * The early stop strategy. If it is null, then early stop is disabled
     */
    private AbstractEarlyStopStrategy earlyStopStrategy;

    /**
     * The model set candidate variables or not
     */
    protected boolean hasCandidates = false;

    /**
     * The Column ID set that is used to build model
     *      - if there are final selected variables, it is the set of all final selected variables
     *      - if there is not final selected variables, it is the set of all *GOOD* variables (for SE)
     */
    protected Set<Integer> modelFeatureSet;

    /**
     * The input vector length for model
     */
    protected int modelInputCnt = 0;

    @Override
    public void init(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        loadConfigFiles(context.getProps());

        // get model input feature and count
        this.modelFeatureSet = DTrainUtils.getModelFeatureSet(this.columnConfigList, this.hasCandidates);
        this.modelInputCnt = DTrainUtils.getFeatureInputsCnt(this.modelConfig, this.columnConfigList, this.modelFeatureSet);

        int trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));

        GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                modelConfig.getTrain().getGridConfigFileContent());
        validParams = this.modelConfig.getTrain().getParams();
        if(gs.hasHyperParam()) {
            validParams = gs.getParams(trainerId);
            LOG.info("Start grid search master with params: {}", validParams);
        }

        this.learningRate = Double.valueOf(this.validParams.get(CommonConstants.LEARNING_RATE).toString());
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
        this.propagation = pObject == null ? "R" : (String) pObject;

        Object rconstant = validParams.get(CommonConstants.REGULARIZED_CONSTANT);
        this.regularizedConstant = NumberFormatUtils.getDouble(rconstant == null ? "" : rconstant.toString(), 0d);

        this.isContinuousEnabled = Boolean.TRUE.toString()
                .equalsIgnoreCase(context.getProps().getProperty(CommonConstants.CONTINUOUS_TRAINING));

        // not initialized and not first iteration, should be fault tolerance, recover state in LogisticRegressionMaster
        this.weights = recoverMasterState(context).getParameters();
    }

    private LogisticRegressionParams initModelParams(LR loadModel) {
        LogisticRegressionParams params = new LogisticRegressionParams();
        params.setTrainError(0);
        params.setValidationError(0);
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
            LR existingModel = (LR) ModelSpecLoaderUtils.loadModel(modelConfig, modelPath,
                    ShifuFileUtils.getFileSystemBySourceType(this.modelConfig.getDataSet().getSource(), modelPath));
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
    public LogisticRegressionParams doCompute(
            MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        // 0. fault tolerance logic for master
        if(isInitialized.compareAndSet(false, true)) {
            // not initialized and not first iteration, should be fault tolerance, recover state in master
            return recoverMasterState(context);
        }

        // 1. init or read weights from existing model if continuous enabled only in first step
        if(context.isFirstIteration()) {
            return initOrContinueTrain(context);
        }

        // 2. accumulate all gradients together
        double[] gradients = new double[this.modelInputCnt + 1]; // append bias
        double trainError = 0.0d, validationError = 0d;
        double trainSize = 0, vldSize = 0, trainCount = 0, vldCount = 0;
        for(LogisticRegressionParams param: context.getWorkerResults()) {
            if(param == null) {
                continue;
            }
            for(int i = 0; i < gradients.length; i++) {
                gradients[i] += param.getParameters()[i];
            }
            trainError += param.getTrainError();
            validationError += param.getValidationError();
            trainSize += param.getTrainSize();
            vldSize += param.getValidationSize();
            trainCount += param.getTrainCount();
            vldCount += param.getValidationCount();
        }

        // 3. compute to get latest model weights; on demand init Weight instance because of trainCount needed
        initWeightOptimizerIfNeeded(trainSize);
        int currItr = context.getCurrentIteration();
        this.weights = this.weightCalculator.calculateWeights(this.weights, gradients, (currItr - 1));

        // 4. return latest model weights to workers
        double finalTrainError = trainError / trainSize;
        double finalTestError = validationError / vldSize;
        LOG.info("Iteration {} with train error {}, test error {}", currItr, finalTrainError, finalTestError);
        return buildReturnParams(context, trainSize, vldSize, trainCount, vldCount, finalTrainError, finalTestError);
    }

    private LogisticRegressionParams buildReturnParams(
            MasterContext<LogisticRegressionParams, LogisticRegressionParams> context, double trainSize,
            double validationSize, double trainCount, double validationCount, double finalTrainError,
            double finalTestError) {
        LogisticRegressionParams lrParams = new LogisticRegressionParams(weights, finalTrainError, finalTestError,
                trainSize, validationSize, trainCount, validationCount);

        if(finalTestError < this.bestValidationError) {
            this.bestValidationError = finalTestError;
        }

        if(earlyStopStrategy != null) {
            boolean isToStopEarly = earlyStopStrategy.shouldEarlyStop(context.getCurrentIteration(), weights,
                    finalTrainError, finalTestError);
            if(isToStopEarly) {
                lrParams.setHalt(true);
            }
        }
        return lrParams;
    }

    private LogisticRegressionParams initOrContinueTrain(
            MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(this.isContinuousEnabled) {
            return initOrRecoverParams(context);
        } else {
            return initWeights();
        }
    }

    private LogisticRegressionParams recoverMasterState(
            MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
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
        return initWeights();
    }

    private void initWeightOptimizerIfNeeded(double trainSize) {
        if(this.weightCalculator == null) {
            this.weightCalculator = new Weight(weights.length, trainSize, learningRate, this.propagation,
                    this.regularizedConstant, RegulationLevel.to(this.validParams.get(CommonConstants.REG_LEVEL_KEY)));
        } else {
            this.weightCalculator.setNumTrainSize(trainSize);
        }
    }

    private LogisticRegressionParams initWeights() {
        weights = new double[this.modelInputCnt + 1];
        for(int i = 0; i < weights.length; i++) {
            weights[i] = nextDouble(-1, 1);
        }
        return new LogisticRegressionParams(weights);
    }

    private void loadConfigFiles(final Properties props) {
        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
            this.hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public final double nextDouble(final double min, final double max) {
        final double range = max - min;
        return (range * Math.random()) + min;
    }

}
