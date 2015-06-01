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
package ml.shifu.shifu.core.validator;

import ml.shifu.shifu.container.meta.MetaFactory;
import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * ModelInspector class is to do Safety Testing for model.
 * <p/>
 * Safety Testing include: 1. validate the @ModelConfig against its meta data
 * 
 * @links{src/main/resources/store/ModelConfigMeta.json 2. check source data for training and evaluation 3. check the
 *                                                      prerequisite for each step
 */
public class ModelInspector {

    public static enum ModelStep {
        INIT, STATS, VARSELECT, NORMALIZE, TRAIN, POSTTRAIN, EVAL
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
     */
    public ValidateResult probe(ModelConfig modelConfig, ModelStep modelStep) throws Exception {
        ValidateResult result = checkMeta(modelConfig);
        if(!result.getStatus()) {
            return result;
        }

        if(modelConfig.getDataSet().getSource() == SourceType.LOCAL
                && modelConfig.getBasic().getRunMode() == RunMode.mapred) {
            ValidateResult tmpResult = new ValidateResult(true);
            // tmpResult.setStatus(false);
            // tmpResult.getCauses().add(
            // "'LOCAL' data set (dataSet.source) cannot be run with 'mapred' run mode(basic.runMode)");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(modelConfig.getDataSet().getAutoType()) {
            if(modelConfig.getDataSet().getAutoTypeThreshold() < 1) {
                ValidateResult tmpResult = new ValidateResult(true);
                // tmpResult.getCauses().add(
                tmpResult.addCause("'autoTypeThreshold' should not be less than 1.");
                result = ValidateResult.mergeResult(result, tmpResult);
            }
        }

        if(ModelStep.INIT.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkTrainData(modelConfig.getDataSet()));
            result = ValidateResult.mergeResult(result, checkVarSelect(modelConfig.getVarSelect()));
            if(result.getStatus()) {
                result = ValidateResult.mergeResult(result, checkColumnConf(modelConfig));
            }
        } else if(ModelStep.STATS.equals(modelStep)) {
            result = ValidateResult.mergeResult(result,
                    checkFile("ColumnConfig.json", SourceType.LOCAL, "ColumnConfig.json : "));
        } else if(ModelStep.VARSELECT.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkVarSelect(modelConfig.getVarSelect()));
            if(result.getStatus()) {
                // user may add configure file between steps
                // add validation to avoid user to make mistake
                result = ValidateResult.mergeResult(result, checkColumnConf(modelConfig));
            }
        } else if(ModelStep.NORMALIZE.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkNormSetting(modelConfig.getNormalize()));
        } else if(ModelStep.TRAIN.equals(modelStep)) {
            result = ValidateResult.mergeResult(result, checkTrainSetting(modelConfig.getTrain()));
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
    private ValidateResult checkVarSelect(ModelVarSelectConf varSelect) throws IOException {
        ValidateResult result = new ValidateResult(true);

        if(Boolean.TRUE.equals(varSelect.getForceEnable())) {
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
                                "forceSelect columns configuration"));
            }
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
        result = ValidateResult.mergeResult(result,
                checkFile(dataSet.getHeaderPath(), dataSet.getSource(), prefix + "header path "));

        return result;
    }

    /**
     * Check the training data for model
     * Fist of all, it checks the @RawSourceData
     * Then, it checks conf file for categorical column exists or not, if the setting is not empty
     * Then, it checks conf file for meta column exists or not, if the setting is not empty
     * 
     * @param dataSet
     *            - @ModelSourceDataConf to check
     * @return @ValidateResult
     * @throws IOException
     *             IOException may be thrown when checking file
     */
    private ValidateResult checkTrainData(ModelSourceDataConf dataSet) throws IOException {
        ValidateResult result = checkRawData(dataSet, "Train Set:");

        if(StringUtils.isNotBlank(dataSet.getCategoricalColumnNameFile())) {
            result = ValidateResult.mergeResult(
                    result,
                    checkFile(dataSet.getCategoricalColumnNameFile(), SourceType.LOCAL,
                            "categorical columns configuration "));
        }

        if(StringUtils.isNotBlank(dataSet.getMetaColumnNameFile())) {
            result = ValidateResult.mergeResult(result,
                    checkFile(dataSet.getMetaColumnNameFile(), SourceType.LOCAL, "meta columns configuration "));
        }

        return result;
    }

    /**
     * Check the setting for model normalize.
     * It will make sure the following condition:
     * 
     * <p>
     * <ul>
     *     <li>stdDevCutOff > 0</li>
     *     <li>0 < sampleRate <= 1</li>
     *     <li>sampleNegOnly is either true or false</li>
     *     <li>normType contains valid value among [ZSCALE, WOE, WEIGHT_WOE, HYBRID, WEIGHT_HYBRID]</li>
     * </ul>
     * </p>
     * 
     * @param norm {@link ModelNormalizeConf} instance.
     * @return check result instance {@link ValidateResult}. 
     */
    private ValidateResult checkNormSetting(ModelNormalizeConf norm) {
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
            tmpResult.getCauses().add("normType should be one of [ZSCALE, WOE, WEIGHT_WOE, HYBRID, WEIGHT_HYBRID] in normalize configuration");
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
    private ValidateResult checkTrainSetting(ModelTrainConf train) {
        ValidateResult result = new ValidateResult(true);

        if(train.getBaggingNum() == null || train.getBaggingNum() < 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("Bagging number should be greater than zero in train configuration");
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

        if(train.getConvergenceThreshold() != null && train.getConvergenceThreshold().compareTo(0.0) < 0) {
            ValidateResult tmpResult = new ValidateResult(true);
            tmpResult.setStatus(false);
            tmpResult.getCauses().add("'threshold' should be large than or equal to 0.0 if set.");
            result = ValidateResult.mergeResult(result, tmpResult);
        }

        if(train.getAlgorithm().equalsIgnoreCase("nn")) {
            Map<String, Object> params = train.getParams();
            int layerCnt = (Integer) params.get(NNTrainer.NUM_HIDDEN_LAYERS);
            if(layerCnt < 0) {
                ValidateResult tmpResult = new ValidateResult(true);
                tmpResult.setStatus(false);
                tmpResult.getCauses().add("The number of hidden layers should be >= 0 in train configuration");
                result = ValidateResult.mergeResult(result, tmpResult);
            }

            List<Integer> hiddenNode = (List<Integer>) params.get(NNTrainer.NUM_HIDDEN_NODES);
            List<String> activateFucs = (List<String>) params.get(NNTrainer.ACTIVATION_FUNC);

            if(hiddenNode.size() != activateFucs.size() || layerCnt != activateFucs.size()) {
                ValidateResult tmpResult = new ValidateResult(true);
                tmpResult.setStatus(false);
                tmpResult.getCauses().add(
                        NNTrainer.NUM_HIDDEN_LAYERS + "/SIZE(" + NNTrainer.NUM_HIDDEN_NODES + ")" + "/SIZE("
                                + NNTrainer.ACTIVATION_FUNC + ")" + " should be equal in train configuration");
                result = ValidateResult.mergeResult(result, tmpResult);
            }

            Double learningRate = Double.valueOf(params.get(NNTrainer.LEARNING_RATE).toString());

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
