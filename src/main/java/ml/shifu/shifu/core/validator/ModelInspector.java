/*
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
package ml.shifu.shifu.core.validator;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.meta.MetaFactory;
import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningAlgorithm;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.container.obj.ModelTrainConf.MultipleClassification;
import ml.shifu.shifu.container.obj.ModelVarSelectConf;
import ml.shifu.shifu.container.obj.ModelVarSelectConf.PostCorrelationMetric;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.FeatureSubsetStrategy;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ModelInspector class is to do Safety Testing for model.
 * 
 * <p>
 * Safety Testing include: 1. validate the ModelConfig against its meta data
 * src/main/resources/store/ModelConfigMeta.json 2. check source data for training and evaluation 3. check the
 * prerequisite for each step
 */
public class ModelInspector {

    private static final Logger LOG = LoggerFactory.getLogger(ModelInspector.class);

    public static enum ModelStep {
        INIT, STATS, VARSELECT, NORMALIZE, TRAIN, POSTTRAIN, EVAL, EXPORT, COMBO
    }

    private static ModelInspector instance = new ModelInspector();

    // singleton class, avoid to create new instance
    private ModelInspector() {
    }

    /**
     * @return the inspector handler
     */
    public static ModelInspector getInspector() {
        return instance;
    }

    /**
     * Probe the status of model for each step.
     * It will check the setting in @ModelConfig to make sure all setting from user are correct.
     * After that it will do different checking for different steps
     * 
     * @param modelConfig
     *            - the model configuration that want to probe
     * @param modelStep
     *            - the steps
     * @return the result of probe
     *         if everything is OK, the status of ValidateResult is TRUE
     *         else the status of ValidateResult is FALSE, and the reasons will in the clauses of ValidateResult
     * @throws Exception
     *             any exception in validation
     */
    public ValidateResult probe(ModelConfig modelConfig, ModelStep modelStep) throws Exception {
        ValidateResult result = checkMeta(modelConfig);
        if(!result.getStatus()) {
            return result;
        }

        if(modelConfig.isClassification()) {
            if(modelConfig.getBasic().getRunMode() == RunMode.LOCAL
                    || modelConfig.getDataSet().getSource() == SourceType.LOCAL) {
                ValidateResult tmpResult = new ValidateResult(true);
                tmpResult.addCause("Multiple classification is only effective in MAPRED runmode and HDFS source type.");
                result = ValidateResult.mergeResult(result, tmpResult);
            }
        }

        if(modelConfig.getDataSet().getSource() == SourceType.LOCAL && modelConfig.isMapReduceRunMode()) {
            ValidateResult tmpResult = new ValidateResult(true);
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(ModelStep.INIT.equals(modelStep)) {
            // in INIT, only check if data or header are there or not
            result = ValidateResult.mergeResult(result, checkRawData(modelConfig.getDataSet(), "Train Set:"));
        } else if(ModelStep.STATS.equals(modelStep)) {
            result = ValidateResult.mergeResult(result,
                    checkFile("ColumnConfig.json", SourceType.LOCAL, "ColumnConfig.json : "));
            result = ValidateResult.mergeResult(result, checkStatsConf(modelConfig));
            // verify categorical name file
            if(StringUtils.isNotBlank(modelConfig.getDataSet().getCategoricalColumnNameFile())) {
                result = ValidateResult.mergeResult(
                        result,
                        checkFile(modelConfig.getDataSet().getCategoricalColumnNameFile(), SourceType.LOCAL,
                                "categorical columns configuration "));
            }

            // verify meta name file
            if(StringUtils.isNotBlank(modelConfig.getDataSet().getMetaColumnNameFile())) {
                result = ValidateResult.mergeResult(
                        result,
                        checkFile(modelConfig.getDataSet().getMetaColumnNameFile(), SourceType.LOCAL,
                                "meta columns configuration "));
            }
            // check column stats
            if(result.getStatus()) {
                result = ValidateResult.mergeResult(result, checkColumnConf(modelConfig));
            }
        } else if(ModelStep.VARSELECT.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkVarSelect(modelConfig, modelConfig.getVarSelect()));
            if(result.getStatus()) {
                // user may add configure file between steps
                // add validation to avoid user to make mistake
                result = ValidateResult.mergeResult(result, checkColumnConf(modelConfig));
            }
        } else if(ModelStep.NORMALIZE.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkNormSetting(modelConfig, modelConfig.getNormalize()));
        } else if(ModelStep.TRAIN.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkTrainSetting(modelConfig, modelConfig.getTrain()));
            if(modelConfig.isClassification()
                    && modelConfig.getTrain().getMultiClassifyMethod() == MultipleClassification.NATIVE) {
                if(!"nn".equalsIgnoreCase((modelConfig.getTrain().getAlgorithm()))
                        && !CommonConstants.RF_ALG_NAME.equalsIgnoreCase(modelConfig.getTrain().getAlgorithm())) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult
                            .addCause("Native multiple classification is only effective in neural network (nn) or random forest (rf) training method.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }
            }

            if(modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll()) {
                if(!CommonUtils.isTreeModel(modelConfig.getAlgorithm())
                        && !modelConfig.getAlgorithm().equalsIgnoreCase("nn")) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult
                            .addCause("OneVSAll multiple classification is only effective in gradient boosted trees (GBT) or random forest (RF) or Neural Network (NN) training method.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }
            }

        } else if(ModelStep.POSTTRAIN.equals(modelStep)) {
            // TODO
        } else if(ModelStep.EVAL.equals(modelStep)) {
            if(CollectionUtils.isNotEmpty(modelConfig.getEvals())) {
                for(EvalConfig evalConfig: modelConfig.getEvals()) {
                    result = ValidateResult.mergeResult(result,
                            checkRawData(evalConfig.getDataSet(), "Eval Set - " + evalConfig.getName() + ": "));
                    if(StringUtils.isNotBlank(evalConfig.getScoreMetaColumnNameFile())) {
                        result = ValidateResult.mergeResult(
                                result,
                                checkFile(evalConfig.getScoreMetaColumnNameFile(), SourceType.LOCAL, "Eval Set - "
                                        + evalConfig.getName() + ": "));
                    }
                }
            }
        }

        return result;
    }

    /**
     * Check the settings in @ModelConfig against the constrains in @MetaFactory
     * 
     * @param modelConfig
     *            - model configuration to check
     * @return - the validation result
     * @throws Exception
     *             Exception when checking model configuration
     */
    public ValidateResult checkMeta(ModelConfig modelConfig) throws Exception {
        return MetaFactory.validate(modelConfig);
    }

    /**
     * Check the target column in @ModelConfit, it shouldn' be null or empty
     * - the target column shouldn't be meta column
     * - the target column shouldn't be force select column
     * - the target column shouldn't be force remove column
     * <p/>
     * - a column shouldn't exist in more than list - metaColumns, forceSelectColumns, forceRemoveColumns
     * 
     * @param modelConfig
     *            - model configuration to check
     * @return - the validation result
     * @throws IOException
     *             Exception when checking model configuration
     */
    private ValidateResult checkColumnConf(ModelConfig modelConfig) throws IOException {
        ValidateResult result = new ValidateResult(true);

        if(StringUtils.isBlank(modelConfig.getTargetColumnName())) {
            result.addCause("The target column name is null or empty.");
        } else {
            List<String> metaColumns = modelConfig.getMetaColumnNames();
            List<String> forceRemoveColumns = modelConfig.getListForceRemove();
            List<String> forceSelectColumns = modelConfig.getListForceSelect();

            if(CollectionUtils.isNotEmpty(metaColumns) && metaColumns.contains(modelConfig.getTargetColumnName())) {
                result.addCause("The target column name shouldn't be in the meta column conf.");
            }

            if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                    && CollectionUtils.isNotEmpty(forceRemoveColumns)
                    && forceRemoveColumns.contains(modelConfig.getTargetColumnName())) {
                result.addCause("The target column name shouldn't be in the force remove conf.");
            }

            if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())
                    && CollectionUtils.isNotEmpty(forceSelectColumns)
                    && forceSelectColumns.contains(modelConfig.getTargetColumnName())) {
                result.addCause("The target column name shouldn't be in the force select conf.");
            }

            if(Boolean.TRUE.equals(modelConfig.getVarSelect().getForceEnable())) {
                String columnColumn = CommonUtils.containsAny(metaColumns, forceRemoveColumns);
                if(columnColumn != null) {
                    result.addCause("Column - " + columnColumn
                            + " exists both in meta column conf and force remove conf.");
                }

                columnColumn = CommonUtils.containsAny(metaColumns, forceSelectColumns);
                if(columnColumn != null) {
                    result.addCause("Column - " + columnColumn
                            + " exists both in meta column conf and force select conf.");
                }

                columnColumn = CommonUtils.containsAny(forceSelectColumns, forceRemoveColumns);
                if(columnColumn != null) {
                    result.addCause("Column - " + columnColumn
                            + " exists both in force select conf and force remove conf.");
                }
            }
        }

        return result;
    }

    private ValidateResult checkStatsConf(ModelConfig modelConfig) throws IOException {
        ValidateResult result = new ValidateResult(true);

        if(modelConfig.isClassification()
                && (modelConfig.getBinningMethod() == BinningMethod.EqualPositive
                        || modelConfig.getBinningMethod() == BinningMethod.EqualNegtive
                        || modelConfig.getBinningMethod() == BinningMethod.WeightEqualPositive || modelConfig
                        .getBinningMethod() == BinningMethod.WeightEqualNegative)) {
            ValidateResult tmpResult = new ValidateResult(false,
                    Arrays.asList("Multiple classification cannot leverage EqualNegtive and EqualPositive binning."));
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(modelConfig.isClassification() && modelConfig.getBinningAlgorithm() != BinningAlgorithm.SPDTI) {
            result = ValidateResult.mergeResult(
                    result,
                    new ValidateResult(false, Arrays
                            .asList("Only SPDTI binning algorithm are supported with multiple classification.")));

        }

        // maxNumBin should be less than Short.MAX_VALUE, larger maxNumBin need more computing and no meaningful.
        if(modelConfig.getStats().getMaxNumBin() > Short.MAX_VALUE || modelConfig.getStats().getMaxNumBin() < 0) {
            result = ValidateResult.mergeResult(result,
                    new ValidateResult(false, Arrays.asList("stats#maxNumBin should be in [0, 32767].")));

        }
        return result;
    }

    /**
     * Check the prerequisite for variable selection
     * 1. if the force remove is not empty, check the conf file exists or not
     * 2. if the force select is not empty, check the conf file exists or not
     * 
     * @param varSelect
     *            - @ModelVarSelectConf settings for variable selection
     * @return - the result of validation
     * @throws IOException
     *             IOException may be thrown when checking file
     */
    private ValidateResult checkVarSelect(ModelConfig modelConfig, ModelVarSelectConf varSelect) throws IOException {
        ValidateResult result = new ValidateResult(true);

        if(Boolean.TRUE.equals(varSelect.getForceEnable())) {
            if(StringUtils.isNotBlank(varSelect.getCandidateColumnNameFile())) {
                result = ValidateResult.mergeResult(
                        result,
                        checkFile(varSelect.getCandidateColumnNameFile(), SourceType.LOCAL,
                                "candidate columns configuration "));
            }
            if(StringUtils.isNotBlank(varSelect.getForceRemoveColumnNameFile())) {
                result = ValidateResult.mergeResult(
                        result,
                        checkFile(varSelect.getForceRemoveColumnNameFile(), SourceType.LOCAL,
                                "forceRemove columns configuration "));
            }

            if(StringUtils.isNotBlank(varSelect.getForceSelectColumnNameFile())) {
                result = ValidateResult.mergeResult(
                        result,
                        checkFile(varSelect.getForceSelectColumnNameFile(), SourceType.LOCAL,
                                "forceSelect columns configuration "));
            }
        }

        PostCorrelationMetric corrMetric = varSelect.getPostCorrelationMetric();
        if(!varSelect.getFilterBy().equals("SE") && corrMetric != null && corrMetric == PostCorrelationMetric.SE) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add(
                    "VarSelect#filterBy and VarSelect#postCorrelationMetric should be both set to SE.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        return result;
    }

    /**
     * Check the Data Set - to check the data exists or not
     * to check the header of data exists or not
     * 
     * @param dataSet
     *            - @RawSourceData to check
     * @param prefix
     *            - the prefix to generate readable clauses
     * @return @ValidateResult
     * @throws IOException
     *             IOException may be thrown when checking file
     */
    private ValidateResult checkRawData(RawSourceData dataSet, String prefix) throws IOException {
        ValidateResult result = new ValidateResult(true);

        result = ValidateResult.mergeResult(result,
                checkFile(dataSet.getDataPath(), dataSet.getSource(), prefix + "data path "));
        if(!StringUtils.isBlank(dataSet.getHeaderPath())) {
            result = ValidateResult.mergeResult(result,
                    checkFile(dataSet.getHeaderPath(), dataSet.getSource(), prefix + "header path "));
        } else {
            LOG.warn("Header file is set to empty, shifu will try to detect schema by first line of input and header "
                    + "delimiter.");
        }
        return result;
    }

    /**
     * Check the setting for model normalize.
     * It will make sure the following condition:
     * 
     * <p>
     * <ul>
     * <li>stdDevCutOff > 0</li>
     * <li>0 < sampleRate <= 1</li>
     * <li>sampleNegOnly is either true or false</li>
     * <li>normType contains valid value among [ZSCALE, WOE, WEIGHT_WOE, HYBRID, WEIGHT_HYBRID]</li>
     * </ul>
     * </p>
     * 
     * @param norm
     *            {@link ModelNormalizeConf} instance.
     * @return check result instance {@link ValidateResult}.
     */
    private ValidateResult checkNormSetting(ModelConfig modelConfig, ModelNormalizeConf norm) {
        ValidateResult result = new ValidateResult(true);

        if(norm.getStdDevCutOff() == null || norm.getStdDevCutOff() <= 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("stdDevCutOff should be positive value in normalize configuration");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(norm.getSampleRate() == null || norm.getSampleRate() <= 0 || norm.getSampleRate() > 1) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("sampleRate should be positive value in normalize configuration");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(norm.getSampleNegOnly() == null) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("sampleNegOnly should be true/false in normalize configuration");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(norm.getNormType() == null) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult
                    .getCauses()
                    .add("normType should be one of [ZSCALE, WOE, WEIGHT_WOE, HYBRID, WEIGHT_HYBRID] in normalize configuration");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        boolean isZScore = modelConfig.getNormalize().getNormType() == NormType.ZSCALE
                || modelConfig.getNormalize().getNormType() == NormType.ZSCORE
                || modelConfig.getNormalize().getNormType() == NormType.OLD_ZSCALE
                || modelConfig.getNormalize().getNormType() == NormType.OLD_ZSCORE
                || modelConfig.getNormalizeType().equals(NormType.ZSCALE_ONEHOT);

        if(modelConfig.isClassification() && !isZScore) {
            ValidateResult tmpResult = new ValidateResult(false);
            tmpResult.getCauses().add(
                    "NormType 'ZSCALE|ZSCORE|ZSCALE_ONEHOT' is the only norm type for multiple classification.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        return result;
    }

    /**
     * Check the setting for model training.
     * It will make sure (num_of_layers > 0
     * && num_of_layers = hidden_nodes_size
     * && num_of_layse = active_func_size)
     * 
     * @param train
     *            - @ModelTrainConf to check
     * @return @ValidateResult
     */
    @SuppressWarnings("unchecked")
    private ValidateResult checkTrainSetting(ModelConfig modelConfig, ModelTrainConf train) {
        ValidateResult result = new ValidateResult(true);

        if(train.getBaggingNum() == null || train.getBaggingNum() < 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("Bagging number should be greater than zero in train configuration");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getNumKFold() != null && train.getNumKFold() > 20) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("numKFold should be in (0, 20] or <=0 (not dp k-crossValidation)");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getBaggingSampleRate() == null || train.getBaggingSampleRate().compareTo(Double.valueOf(0)) <= 0
                || train.getBaggingSampleRate().compareTo(Double.valueOf(1)) > 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("Bagging sample rate number should be in (0, 1].");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getValidSetRate() == null || train.getValidSetRate().compareTo(Double.valueOf(0)) < 0
                || train.getValidSetRate().compareTo(Double.valueOf(1)) >= 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("Validation set rate number should be in [0, 1).");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getNumTrainEpochs() == null || train.getNumTrainEpochs() <= 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("Epochs should be larger than 0.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getEpochsPerIteration() != null && train.getEpochsPerIteration() <= 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("'epochsPerIteration' should be larger than 0 if set.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getWorkerThreadCount() != null
                && (train.getWorkerThreadCount() <= 0 || train.getWorkerThreadCount() > 32)) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("'workerThreadCount' should be in (0, 32] if set.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getConvergenceThreshold() != null && train.getConvergenceThreshold().compareTo(0.0) < 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("'threshold' should be larger than or equal to 0.0 if set.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(modelConfig.isClassification() && train.isOneVsAll() && !CommonUtils.isTreeModel(train.getAlgorithm())
                && !train.getAlgorithm().equalsIgnoreCase("nn")) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add(
                    "'one vs all' or 'one vs rest' is only enabled with 'RF' or 'GBT' or 'NN' algorithm");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(modelConfig.isClassification() && train.getMultiClassifyMethod() == MultipleClassification.NATIVE
                && train.getAlgorithm().equalsIgnoreCase(CommonConstants.RF_ALG_NAME)) {
            Object impurity = train.getParams().get("Impurity");
            if(impurity != null && !"entropy".equalsIgnoreCase(impurity.toString())
                    && !"gini".equalsIgnoreCase(impurity.toString())) {
                ValidateResult tmpResult = new ValidateResult(true);
                tmpResult.setStatus(false);
                tmpResult.getCauses().add(
                        "Impurity should be in [entropy,gini] if native mutiple classification in RF.");
                result = ValidateResult.mergeResult(result, tmpResult);
            }
        }

        GridSearch gs = new GridSearch(train.getParams(), train.getGridConfigFileContent());
        // such parameter validation only in regression and not grid search mode
        if(modelConfig.isRegression() && !gs.hasHyperParam()) {
            if(train.getAlgorithm().equalsIgnoreCase("nn")) {
                Map<String, Object> params = train.getParams();

                Object loss = params.get("Loss");
                if(loss != null && !"log".equalsIgnoreCase(loss.toString())
                        && !"squared".equalsIgnoreCase(loss.toString())
                        && !"absolute".equalsIgnoreCase(loss.toString())) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add("Loss should be in [log,squared,absolute].");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                int layerCnt = (Integer) params.get(CommonConstants.NUM_HIDDEN_LAYERS);
                if(layerCnt < 0) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add("The number of hidden layers should be >= 0 in train configuration");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                List<Integer> hiddenNode = (List<Integer>) params.get(CommonConstants.NUM_HIDDEN_NODES);
                List<String> activateFucs = (List<String>) params.get(CommonConstants.ACTIVATION_FUNC);

                if(hiddenNode.size() != activateFucs.size() || layerCnt != activateFucs.size()) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add(
                            CommonConstants.NUM_HIDDEN_LAYERS + "/SIZE(" + CommonConstants.NUM_HIDDEN_NODES + ")"
                                    + "/SIZE(" + CommonConstants.ACTIVATION_FUNC + ")"
                                    + " should be equal in train configuration");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Double learningRate = Double.valueOf(params.get(CommonConstants.LEARNING_RATE).toString());
                if(learningRate != null && (learningRate.compareTo(Double.valueOf(0)) <= 0)) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add("Learning rate should be larger than 0.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Object learningDecayO = params.get("LearningDecay");
                if(learningDecayO != null) {
                    Double learningDecay = Double.valueOf(learningDecayO.toString());
                    if(learningDecay != null
                            && ((learningDecay.compareTo(Double.valueOf(0)) < 0) || (learningDecay.compareTo(Double
                                    .valueOf(1)) >= 0))) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("Learning decay should be in [0, 1) if set.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object elmObject = params.get(DTrainUtils.IS_ELM);
                boolean isELM = elmObject == null ? false : "true".equalsIgnoreCase(elmObject.toString());
                if(isELM && layerCnt != 1) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add(
                            "If ELM(extreme learning machine), hidden layer should only be one layer.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Object dropoutObj = params.get(CommonConstants.DROPOUT_RATE);
                if(dropoutObj != null) {
                    Double dropoutRate = Double.valueOf(dropoutObj.toString());
                    if(dropoutRate != null && (dropoutRate < 0d || dropoutRate >= 1d)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("Dropout rate should be in [0, 1).");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object miniBatchsO = params.get("MiniBatchs");
                if(miniBatchsO != null) {
                    Integer miniBatchs = Integer.valueOf(miniBatchsO.toString());
                    if(miniBatchs != null && (miniBatchs <= 0 || miniBatchs > 1000)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("MiniBatchs should be in (0, 1000] if set.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object momentumO = params.get("Momentum");
                if(momentumO != null) {
                    Double momentum = Double.valueOf(momentumO.toString());
                    if(momentum != null && momentum <= 0d) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("Momentum should be in (0, ) if set.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object adamBeta1O = params.get("AdamBeta1");
                if(adamBeta1O != null) {
                    Double adamBeta1 = Double.valueOf(adamBeta1O.toString());
                    if(adamBeta1 != null && (adamBeta1 <= 0d || adamBeta1 >= 1d)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("AdamBeta1 should be in (0, 1) if set.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object adamBeta2O = params.get("AdamBeta2");
                if(adamBeta2O != null) {
                    Double adamBeta2 = Double.valueOf(adamBeta2O.toString());
                    if(adamBeta2 != null && (adamBeta2 <= 0d || adamBeta2 >= 1d)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("AdamBeta2 should be in (0, 1) if set.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }
            }

            if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.GBT_ALG_NAME)
                    || train.getAlgorithm().equalsIgnoreCase(CommonConstants.RF_ALG_NAME)
                    || train.getAlgorithm().equalsIgnoreCase(NNConstants.NN_ALG_NAME)) {
                Map<String, Object> params = train.getParams();
                Object fssObj = params.get("FeatureSubsetStrategy");

                if(fssObj == null) {
                    if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.GBT_ALG_NAME)
                            || train.getAlgorithm().equalsIgnoreCase(CommonConstants.RF_ALG_NAME)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("FeatureSubsetStrategy is not set in RF/GBT algorithm.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                } else {
                    boolean isNumber = false;
                    double doubleFss = 0;
                    try {
                        doubleFss = Double.parseDouble(fssObj.toString());
                        isNumber = true;
                    } catch (Exception e) {
                        isNumber = false;
                    }

                    if(isNumber) {
                        // if not in [0, 1] failed
                        if(doubleFss <= 0d || doubleFss > 1d) {
                            ValidateResult tmpResult = new ValidateResult(true);
                            tmpResult.setStatus(false);
                            tmpResult.getCauses().add("FeatureSubsetStrategy if double should be in (0, 1]");
                            result = ValidateResult.mergeResult(result, tmpResult);
                        }
                    } else {
                        boolean fssInEnum = false;
                        for(FeatureSubsetStrategy fss: FeatureSubsetStrategy.values()) {
                            if(fss.toString().equalsIgnoreCase(fssObj.toString())) {
                                fssInEnum = true;
                                break;
                            }
                        }

                        if(!fssInEnum) {
                            ValidateResult tmpResult = new ValidateResult(true);
                            tmpResult.setStatus(false);
                            tmpResult
                                    .getCauses()
                                    .add("FeatureSubsetStrategy if string should be in ['ALL', 'HALF', 'ONETHIRD' , 'TWOTHIRDS' , 'AUTO' , 'SQRT' , 'LOG2']");
                            result = ValidateResult.mergeResult(result, tmpResult);
                        }
                    }
                }
            }

            if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.GBT_ALG_NAME)
                    || train.getAlgorithm().equalsIgnoreCase(CommonConstants.RF_ALG_NAME)) {
                Map<String, Object> params = train.getParams();
                if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.GBT_ALG_NAME)) {
                    Object loss = params.get("Loss");
                    if(loss != null && !"log".equalsIgnoreCase(loss.toString())
                            && !"squared".equalsIgnoreCase(loss.toString())
                            && !"halfgradsquared".equalsIgnoreCase(loss.toString())
                            && !"absolute".equalsIgnoreCase(loss.toString())) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("Loss should be in [log,squared,halfgradsquared,absolute].");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }

                    if(loss == null) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add(
                                "'Loss' parameter isn't being set in train#parameters in GBT training.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object maxDepthObj = params.get("MaxDepth");
                if(maxDepthObj != null) {
                    int maxDepth = Integer.valueOf(maxDepthObj.toString());
                    if(maxDepth <= 0 || maxDepth > 20) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("MaxDepth should in [1, 20].");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object vtObj = params.get("ValidationTolerance");
                if(vtObj != null) {
                    double validationTolerance = Double.valueOf(vtObj.toString());
                    if(validationTolerance < 0d || validationTolerance >= 1d) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("ValidationTolerance should in [0, 1).");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object maxLeavesObj = params.get("MaxLeaves");
                if(maxLeavesObj != null) {
                    int maxLeaves = Integer.valueOf(maxLeavesObj.toString());
                    if(maxLeaves <= 0) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("MaxLeaves should in [1, Integer.MAX_VALUE].");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                if(maxDepthObj == null && maxLeavesObj == null) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult
                            .getCauses()
                            .add("'MaxDepth' or 'MaxLeaves' parameters at least one of both should be set in train#parameters in GBT training.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Object maxStatsMemoryMBObj = params.get("MaxStatsMemoryMB");
                if(maxStatsMemoryMBObj != null) {
                    int maxStatsMemoryMB = Integer.valueOf(maxStatsMemoryMBObj.toString());
                    if(maxStatsMemoryMB <= 0) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("MaxStatsMemoryMB should > 0.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object dropoutObj = params.get(CommonConstants.DROPOUT_RATE);
                if(dropoutObj != null) {
                    Double dropoutRate = Double.valueOf(dropoutObj.toString());
                    if(dropoutRate != null && (dropoutRate < 0d || dropoutRate >= 1d)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("Dropout rate should be in [0, 1).");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.GBT_ALG_NAME)) {
                    Object learningRateObj = params.get(CommonConstants.LEARNING_RATE);
                    if(learningRateObj != null) {
                        Double learningRate = Double.valueOf(learningRateObj.toString());
                        if(learningRate != null && (learningRate.compareTo(Double.valueOf(0)) <= 0)) {
                            ValidateResult tmpResult = new ValidateResult(true);
                            tmpResult.setStatus(false);
                            tmpResult.getCauses().add("Learning rate should be larger than 0.");
                            result = ValidateResult.mergeResult(result, tmpResult);
                        }
                    } else {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add(
                                "'LearningRate' parameter isn't being set in train#parameters in GBT training.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                }

                Object minInstancesPerNodeObj = params.get("MinInstancesPerNode");
                if(minInstancesPerNodeObj != null) {
                    int minInstancesPerNode = Integer.valueOf(minInstancesPerNodeObj.toString());
                    if(minInstancesPerNode <= 0) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("MinInstancesPerNode should > 0.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                } else {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add(
                            "'MinInstancesPerNode' parameter isn't be set in train#parameters in GBT/RF training.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Object treeNumObj = params.get("TreeNum");
                if(treeNumObj != null) {
                    int treeNum = Integer.valueOf(treeNumObj.toString());
                    if(treeNum <= 0 || treeNum > 10000) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("TreeNum should be in [1, 10000].");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                } else {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add(
                            "'TreeNum' parameter isn't being set in train#parameters in GBT/RF training.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Object minInfoGainObj = params.get("MinInfoGain");
                if(minInfoGainObj != null) {
                    Double minInfoGain = Double.valueOf(minInfoGainObj.toString());
                    if(minInfoGain != null && (minInfoGain.compareTo(Double.valueOf(0)) < 0)) {
                        ValidateResult tmpResult = new ValidateResult(true);
                        tmpResult.setStatus(false);
                        tmpResult.getCauses().add("MinInfoGain should be >= 0.");
                        result = ValidateResult.mergeResult(result, tmpResult);
                    }
                } else {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add(
                            "'MinInfoGain' parameter isn't be set in train#parameters in GBT/RF training.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                }

                Object impurityObj = params.get("Impurity");
                if(impurityObj == null) {
                    ValidateResult tmpResult = new ValidateResult(true);
                    tmpResult.setStatus(false);
                    tmpResult.getCauses().add("Impurity is not set in RF/GBT algorithm.");
                    result = ValidateResult.mergeResult(result, tmpResult);
                } else {
                    if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.GBT_ALG_NAME)) {
                        if(impurityObj != null && !"variance".equalsIgnoreCase(impurityObj.toString())
                                && !"friedmanmse".equalsIgnoreCase(impurityObj.toString())) {
                            ValidateResult tmpResult = new ValidateResult(true);
                            tmpResult.setStatus(false);
                            tmpResult.getCauses().add("GBDT only supports 'variance|friedmanmse' impurity type.");
                            result = ValidateResult.mergeResult(result, tmpResult);
                        }
                    }

                    if(train.getAlgorithm().equalsIgnoreCase(CommonConstants.RF_ALG_NAME)) {
                        if(impurityObj != null && !"friedmanmse".equalsIgnoreCase(impurityObj.toString())
                                && !"entropy".equalsIgnoreCase(impurityObj.toString())
                                && !"variance".equalsIgnoreCase(impurityObj.toString())
                                && !"gini".equalsIgnoreCase(impurityObj.toString())) {
                            ValidateResult tmpResult = new ValidateResult(true);
                            tmpResult.setStatus(false);
                            tmpResult.getCauses()
                                    .add("RF supports 'variance|entropy|gini|friedmanmse' impurity types.");
                            result = ValidateResult.mergeResult(result, tmpResult);
                        }
                    }
                }
            }

        }
        return result;
    }

    /**
     * check the file exists or not
     * 
     * @param dataPath
     *            - the path of data
     * @param sourceType
     *            - the source type of data [local/hdfs/s3]
     * @param prefix
     *            - the prefix to generate readable clauses
     * @return @ValidateResult
     * @throws IOException
     */
    private ValidateResult checkFile(String dataPath, SourceType sourceType, String prefix) throws IOException {
        ValidateResult result = new ValidateResult(true);

        if(StringUtils.isBlank(dataPath)) {
            result.addCause(prefix + "is null or empty - " + dataPath);
        } else if(dataPath.trim().contains("~")) {
            result.addCause(prefix + "contains ~, which is not allowed - " + dataPath);
        } else if(!ShifuFileUtils.isFileExists(dataPath, sourceType)) {
            result.addCause(prefix + "doesn't exist - " + dataPath);
        }

        return result;
    }
}
