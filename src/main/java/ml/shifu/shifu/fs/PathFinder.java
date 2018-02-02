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
package ml.shifu.shifu.fs;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.impl.util.JarManager;

import java.io.File;
import java.util.Map;

/**
 * {@link PathFinder} is used to obtain all files which can be used in our framework. Some are used for training,
 * evaling, performance ...
 * 
 * <p>
 * {@link #modelConfig} should be passed as parameter in constructor
 */
public class PathFinder {

    public static final String FEATURE_IMPORTANCE_FILE = "all.fi";
    private static final String CORRELATION_CSV = "correlation.csv";
    private static final String REASON_CODE_PATH = "common/ReasonCodeMapV3.json";
    private static final String SHIFU_JAR_PATH = "lib/*.jar";

    /**
     * {@link PathFinder#modelConfig} is used to get global setting for model config path.
     */
    private ModelConfig modelConfig;

    /**
     * If not specified SHIFU_HOME env, some key configurations like pig path or lib path can be configured here
     */
    private Map<String, Object> otherConfigs;

    /**
     * Constructor with valid parameter modelConfig
     * 
     * @param modelConfig
     *            the model config
     * @throws IllegalArgumentException
     *             if {@link #modelConfig} is null.
     */
    public PathFinder(ModelConfig modelConfig) {
        if(modelConfig == null) {
            throw new IllegalArgumentException("modelConfig should not be null.");
        }
        this.modelConfig = modelConfig;
    }

    /**
     * Constructor with valid parameter modelConfig
     * 
     * @param modelConfig
     *            - modelConfig to find
     * @param otherConfigs
     *            other configuration parameters
     * @throws IllegalArgumentException
     *             if {@link #modelConfig} is null.
     */
    public PathFinder(ModelConfig modelConfig, Map<String, Object> otherConfigs) {
        this(modelConfig);
        this.otherConfigs = otherConfigs;
    }

    /**
     * Get absolute path with SHIFU_HOME env.
     * - if the path is already absolute path, just return
     * - or assume it is relative path to SHIFU_HOME
     * 
     * @param path
     *            the given path
     * @return absolute path the absolute path
     */
    public String getScriptPath(String path) {
        String shifuHome = Environment.getProperty(Environment.SHIFU_HOME);
        if(shifuHome == null || shifuHome.length() == 0) {
            // return relative path which is in shifu-*.jar
            return path;
        } else {
            String pathStr = (new Path(path)).isAbsolute() ? path : new Path(
                    Environment.getProperty(Environment.SHIFU_HOME), path).toString();
            File file = new File(pathStr);
            if(file.exists()) {
                return pathStr;
            } else {
                // return arguly path which is in shifu-*.jar
                return path;
            }
        }
    }

    public String getAbsolutePath(String path) {
        String shifuHome = Environment.getProperty(Environment.SHIFU_HOME);
        if(shifuHome == null || shifuHome.length() == 0) {
            // return absolute path which is in shifu-*.jar
            return path;
        } else {
            return (new Path(path)).isAbsolute() ? path : new Path(Environment.getProperty(Environment.SHIFU_HOME),
                    path).toString();
        }
    }

    /**
     * Get project jar file path name.
     * - Since the Jar Path is only used in pig code compiling, just return local path
     * 
     * @return path of SHIFU dependent jars
     */
    public String getJarPath() {
        String shifuHome = Environment.getProperty(Environment.SHIFU_HOME);
        if(shifuHome == null || shifuHome.length() == 0) {
            return new Path(JarManager.findContainingJar(PathFinder.class)).toString();
        } else {
            // very ugly to check if it is outside jar
            try {
                Class.forName("ml.shifu.dtrain.DTrainRequestProcessor");
                return new Path(JarManager.findContainingJar(PathFinder.class)).toString();
            } catch (ClassNotFoundException e) {
                return getAbsolutePath(SHIFU_JAR_PATH);
            }
        }
    }

    /**
     * Get reason code path name.
     * - Reason code file may be used in MapReduce job, so it should have hdfs path
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of reason code file
     */
    public String getReasonCodeMapPath(SourceType sourceType) {
        switch(sourceType) {
            case LOCAL:
                return getAbsolutePath(REASON_CODE_PATH);
            case HDFS:
                return (new Path(getModelSetHdfsPath(), Constants.REASON_CODE_MAP_JSON)).toString();
            default:
                // Others, maybe be we will support S3 in future
                throw new NotImplementedException("Source type - " + sourceType.name() + " is not supported yet!");
        }
    }

    /**
     * Get the file path of ModelConfig
     * 
     * @return path to ModelConfig
     */
    public String getModelConfigPath() {
        return getPathBySourceType(Constants.LOCAL_MODEL_CONFIG_JSON, SourceType.LOCAL);
    }

    /**
     * Get the file path of ModelConfig
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path to ModelConfig
     */
    public String getModelConfigPath(SourceType sourceType) {
        return getPathBySourceType(Constants.MODEL_CONFIG_JSON_FILE_NAME, sourceType);
    }

    /**
     * Get the file path of ColumnConfig
     * 
     * @return path to ColumnConfig
     */
    public String getColumnConfigPath() {
        return getPathBySourceType(Constants.LOCAL_COLUMN_CONFIG_JSON, SourceType.LOCAL);
    }

    /**
     * Get the file path of ColumnConfig
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of ColumnConfig
     */
    public String getColumnConfigPath(SourceType sourceType) {
        return getPathBySourceType(Constants.COLUMN_CONFIG_JSON_FILE_NAME, sourceType);
    }

    public String getLocalColumnStatsPath() {

        return getPathBySourceType(Constants.COLUMN_META_FOLDER_NAME + File.separator
                + Constants.COLUMN_STATS_CSV_FILE_NAME, SourceType.LOCAL);
    }

    /**
     * Get pre-traing stats path.
     * 
     * @return path of pre-training stats file
     */
    public String getPreTrainingStatsPath() {
        return getPreTrainingStatsPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get pre-traing stats path.
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of pre-training stats file
     */
    public String getPreTrainingStatsPath(SourceType sourceType) {
        String preTrainingStatsPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_PRE_TRAIN_STATS_PATH);

        if(StringUtils.isBlank(preTrainingStatsPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.PRE_TRAINING_STATS), sourceType);
        } else {
            return new Path(preTrainingStatsPath).toString();
        }
    }

    public String getStatsSmallBins() {
        return getStatsSmallBins(modelConfig.getDataSet().getSource());
    }

    /**
     * Get stats small bins path
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of stats small-bin file
     */
    public String getStatsSmallBins(SourceType sourceType) {
        return getPathBySourceType(new Path(Constants.TMP, Constants.STATS_SMALL_BINS), sourceType);
    }

    /**
     * Get post train out put path
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of pre-training stats file
     */
    public String getPostTrainOutputPath(SourceType sourceType) {
        String postTrainOutput = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_PRE_TRAIN_STATS_PATH);

        if(StringUtils.isBlank(postTrainOutput)) {
            return getPathBySourceType(new Path(Constants.TMP, "posttrain-output"), sourceType);
        } else {
            return new Path(postTrainOutput).toString();
        }
    }

    /**
     * Get auto type distinct count file path.
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of auto type folder name
     */
    public String getAutoTypeFilePath(SourceType sourceType) {
        String preTrainingStatsPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_AUTO_TYPE_PATH);

        if(StringUtils.isBlank(preTrainingStatsPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.AUTO_TYPE_PATH), sourceType);
        } else {
            return new Path(preTrainingStatsPath).toString();
        }
    }

    /**
     * Get correlation result file path.
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of auto type folder name
     */
    public String getCorrelationPath(SourceType sourceType) {
        String preTrainingStatsPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_CORRELATION_PATH);

        if(StringUtils.isBlank(preTrainingStatsPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.CORRELATION_PATH), sourceType);
        } else {
            return new Path(preTrainingStatsPath).toString();
        }
    }

    /**
     * Get the path for select raw data
     * 
     * @return path of selected raw data
     */
    public String getSelectedRawDataPath() {
        return getSelectedRawDataPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the path for select raw data
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of selected raw data
     */
    public String getSelectedRawDataPath(SourceType sourceType) {
        String selectedRawDataPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_SELECTED_RAW_DATA_PATH);

        if(StringUtils.isBlank(selectedRawDataPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.SELECTED_RAW_DATA), sourceType);
        } else {
            return new Path(selectedRawDataPath).toString();
        }
    }

    /**
     * Get the path of normalized data
     * 
     * @return path of normalized data
     */
    public String getNormalizedDataPath() {
        return getNormalizedDataPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the path of normalized cross validation data
     * 
     * @return path of normalized cross validation data
     */
    public String getNormalizedValidationDataPath() {
        return getNormalizedValidationDataPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the path of normalized data
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of normalized data
     */
    public String getNormalizedDataPath(SourceType sourceType) {
        String normalizedDataPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_NORMALIZED_DATA_PATH);

        if(StringUtils.isBlank(normalizedDataPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.NORMALIZED_DATA), sourceType);
        } else {
            return new Path(normalizedDataPath).toString();
        }
    }

    public String getCleanedDataPath() {
        return getCleanedDataPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Clean and filter raw data set for RF/GBT model inputs
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of normalized data
     */
    public String getCleanedDataPath(SourceType sourceType) {
        String normalizedDataPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_CLEANED_DATA_PATH);

        if(StringUtils.isBlank(normalizedDataPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.CLEANED_DATA), sourceType);
        } else {
            return new Path(normalizedDataPath).toString();
        }
    }

    public String getCleanedValidationDataPath() {
        return getCleanedValidationDataPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Clean and filter raw validation data set for RF/GBT model inputs
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of normalized data
     */
    public String getCleanedValidationDataPath(SourceType sourceType) {
        String normalizedDataPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_CLEANED_VALIDATION_DATA_PATH);

        if(StringUtils.isBlank(normalizedDataPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.CLEANED_VALIDATION_DATA), sourceType);
        } else {
            return new Path(normalizedDataPath).toString();
        }
    }

    /**
     * Get the path of validation normalized data
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of normalized data
     */
    public String getNormalizedValidationDataPath(SourceType sourceType) {
        String normalizedDataPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_NORMALIZED_VALIDATION_DATA_PATH);

        if(StringUtils.isBlank(normalizedDataPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.NORMALIZED_VALIDATION_DATA), sourceType);
        } else {
            return new Path(normalizedDataPath).toString();
        }
    }

    /**
     * Get the path of varselect MSE stats path.
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of var select MSE stats path
     */
    public String getVarSelectMSEOutputPath(SourceType sourceType) {
        String varSelectStatsPath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_VARSLECT_STATS_PATH);

        if(StringUtils.isBlank(varSelectStatsPath)) {
            return getPathBySourceType(new Path(Constants.TMP, "varselectStats"), sourceType);
        } else {
            return new Path(varSelectStatsPath).toString();
        }
    }

    /**
     * Get the path of varselect MSE stats path.
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of var select MSE stats path
     */
    public String getUpdatedBinningInfoPath(SourceType sourceType) {
        String preTrainPath = getPreferPath(modelConfig.getTrain().getCustomPaths(), Constants.KEY_PRE_TRAIN_STATS_PATH);

        if(StringUtils.isBlank(preTrainPath)) {
            return getPathBySourceType(new Path(Constants.TMP, "UpdatedBinningInfo"), sourceType);
        } else {
            return new Path(preTrainPath).toString();
        }
    }

    public String getPSIInfoPath() {
        return this.getPSIInfoPath(modelConfig.getDataSet().getSource());
    }

    public String getPSIInfoPath(SourceType sourceType) {
        String preTrainPath = getPreferPath(modelConfig.getTrain().getCustomPaths(), Constants.KEY_PRE_PSI_PATH);

        if(StringUtils.isBlank(preTrainPath)) {
            return getPathBySourceType(new Path(Constants.TMP, "PSIInfo"), sourceType);
        } else {
            return new Path(preTrainPath).toString();
        }

    }

    /**
     * Get the path of models
     * 
     * @return path of models
     */
    public String getModelsPath() {
        return getModelsPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the path of models
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of models
     */
    public String getModelsPath(SourceType sourceType) {
        return getPathBySourceType(new Path(Constants.MODELS), sourceType);
    }

    /**
     * Get the path of one bagging model
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of models
     */
    public String getBaggingModelPath(SourceType sourceType) {
        return getPathBySourceType(new Path("onebaggingmodel"), sourceType);
    }

    /**
     * Get the path ofnn binary models
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of models
     */
    public String getNNBinaryModelsPath(SourceType sourceType) {
        return getPathBySourceType(new Path("bmodels"), sourceType);
    }

    public String getValErrorPath(SourceType sourceType) {
        return getPathBySourceType(new Path(Constants.TMP, "valerr"), sourceType);
    }

    /**
     * Get the path of models
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of models
     */
    public String getTmpModelsPath(SourceType sourceType) {
        return getPathBySourceType(new Path(Constants.TMP, Constants.DEFAULT_MODELS_TMP_FOLDER), sourceType);
    }

    public String getVarSelsPath(SourceType sourceType) {

        return getPathBySourceType(new Path(Constants.VarSels), sourceType);
    }

    public String getModelVersion(SourceType sourceType) {
        switch(sourceType) {
            case LOCAL:
                return new Path(this.getModelSetLocalPath(), Constants.BACKUPNAME).toString();
            case HDFS:
                return new Path(this.getModelSetHdfsPath(), Constants.BACKUPNAME).toString();
            default:
                // Others, maybe be we will support S3 in future
                throw new NotImplementedException("Source type - " + sourceType.name() + " is not supported yet!");
        }
    }

    /**
     * Get the path of average score for each bin
     * 
     * @return path of bin average score
     */
    public String getBinAvgScorePath() {
        return getBinAvgScorePath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the path of average score for each bin
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of bin average score
     */
    public String getBinAvgScorePath(SourceType sourceType) {
        String binAvgScorePath = getPreferPath(modelConfig.getTrain().getCustomPaths(),
                Constants.KEY_BIN_AVG_SCORE_PATH);

        if(StringUtils.isBlank(binAvgScorePath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.BIN_AVG_SCORE), sourceType);
        } else {
            return new Path(binAvgScorePath).toString();
        }
    }

    /**
     * Get the path of train score
     * 
     * @return path of train score
     */
    public String getTrainScoresPath() {
        return getTrainScoresPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the path of train score
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of train score
     */
    public String getTrainScoresPath(SourceType sourceType) {
        String trainScoresPath = getPreferPath(modelConfig.getTrain().getCustomPaths(), Constants.KEY_TRAIN_SCORES_PATH);

        if(StringUtils.isBlank(trainScoresPath)) {
            return getPathBySourceType(new Path(Constants.TMP, Constants.TRAIN_SCORES), sourceType);
        } else {
            return new Path(trainScoresPath).toString();
        }
    }

    /**
     * Get the evaluation root path according the source type
     * 
     * @param sourceType
     *            - Local/HDFS
     * @return path of evaluation root
     */
    public String getEvalsPath(SourceType sourceType) {
        Path path = new Path(Constants.EVAL_DIR);
        return this.getPathBySourceType(path.toString(), sourceType);
    }

    /**
     * Get evaluation set home directory
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @return path of evaluation set home directory
     */
    public String getEvalSetPath(EvalConfig evalConfig) {
        return getEvalSetPath(evalConfig, evalConfig.getDataSet().getSource());
    }

    /**
     * Get evaluation set home directory, something like Model/eval/evalName)
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param sourceType
     *            - Local/HDFS
     * @return path of evaluation set home directory
     */
    public String getEvalSetPath(EvalConfig evalConfig, SourceType sourceType) {
        return getEvalSetPath(evalConfig.getName(), sourceType);
    }

    /**
     * Get evaluation set home directory, something like eval name
     * 
     * @param evalName
     *            - evalset name to find
     * @param sourceType
     *            - Local/HDFS
     * @return path of evaluation set home directory
     */
    public String getEvalSetPath(String evalName, SourceType sourceType) {
        return new Path(this.getEvalsPath(sourceType), evalName).toString();
    }

    /**
     * Get the path of evaluation normalized data
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @return path of evaluation normalized data
     */
    public String getEvalNormalizedPath(EvalConfig evalConfig) {
        return getEvalNormalizedPath(evalConfig, evalConfig.getDataSet().getSource());
    }

    /**
     * Get the path of evaluation normalized data
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param sourceType
     *            - Local/HDFS
     * @return path of evaluation normalized data
     */
    public String getEvalNormalizedPath(EvalConfig evalConfig, SourceType sourceType) {
        return getEvalFilePath(evalConfig.getName(), Constants.EVAL_NORMALIZED, sourceType);
    }

    /**
     * Get the path of evaluation score
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @return path of evaluation score
     */
    public String getEvalScorePath(EvalConfig evalConfig) {
        return getEvalScorePath(evalConfig, evalConfig.getDataSet().getSource());
    }

    /**
     * Get the path of evaluation score
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param metaColumn
     *            - score column
     * @return path of evaluation score
     */
    public String getEvalMetaScorePath(EvalConfig evalConfig, String metaColumn) {
        SourceType sourceType = evalConfig.getDataSet().getSource();

        String scoreMetaPath = getPreferPath(evalConfig.getCustomPaths(), Constants.KEY_SCORE_PATH);
        if(StringUtils.isBlank(scoreMetaPath)) {
            scoreMetaPath = getEvalFilePath(evalConfig.getName(), Constants.EVAL_META_SCORE, sourceType);
        } else {
            scoreMetaPath = new Path(scoreMetaPath, Constants.EVAL_META_SCORE).toString();
        }

        return new Path(scoreMetaPath, metaColumn).toString();
    }

    /**
     * Get the path of evaluation score
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param sourceType
     *            - Local/HDFS
     * @return path of evaluation score
     */
    public String getEvalScorePath(EvalConfig evalConfig, SourceType sourceType) {
        String scorePath = getPreferPath(evalConfig.getCustomPaths(), Constants.KEY_SCORE_PATH);
        if(StringUtils.isBlank(scorePath)) {
            return getEvalFilePath(evalConfig.getName(), Constants.EVAL_SCORE, sourceType);
        } else {
            return new Path(scorePath).toString();
        }
    }

    public String getEvalConfusionPath(EvalConfig evalConfig, SourceType sourceType) {
        String scorePath = getPreferPath(evalConfig.getCustomPaths(), Constants.KEY_SCORE_PATH);
        if(StringUtils.isBlank(scorePath)) {
            return getEvalFilePath(evalConfig.getName(), Constants.EVAL_SCORE, sourceType);
        } else {
            return new Path(scorePath).toString();
        }
    }

    /**
     * Get the header path of evaluation score
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @return path of evaluation score header
     */
    public String getEvalScoreHeaderPath(EvalConfig evalConfig) {
        return getEvalScoreHeaderPath(evalConfig, evalConfig.getDataSet().getSource());
    }

    /**
     * Get the header path of evaluation score
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param sourceType
     *            - Local/HDFS
     * @return path of evaluation score header
     */
    public String getEvalScoreHeaderPath(EvalConfig evalConfig, SourceType sourceType) {
        String scorePath = getEvalScorePath(evalConfig, sourceType);
        return new Path(scorePath, Constants.PIG_HEADER).toString();
    }

    /**
     * Get the path of evaluation set performance for EvalMetaScore column
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param metaColumn
     *            meta column
     * @return eval meta path
     */
    public String getEvalMetaPerformancePath(EvalConfig evalConfig, String metaColumn) {
        String evalPerformancePath = getPreferPath(evalConfig.getCustomPaths(), Constants.KEY_PERFORMANCE_PATH);

        if(StringUtils.isBlank(evalPerformancePath)) {
            String evalMetaPerfPath = getEvalFilePath(evalConfig.getName(), Constants.EVAL_META_SCORE, evalConfig
                    .getDataSet().getSource());
            return new Path(evalMetaPerfPath, metaColumn + Constants.EVAL_PERFORMANCE).toString();
        } else {
            return new Path(evalPerformancePath, metaColumn + Constants.EVAL_PERFORMANCE).toString();
        }
    }

    /**
     * Get the path of evaluation set performance
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @return path of evaluation set performance
     */
    public String getEvalPerformancePath(EvalConfig evalConfig) {
        return getEvalPerformancePath(evalConfig, evalConfig.getDataSet().getSource());
    }

    /**
     * Get the path of evaluation set performance
     * 
     * @param evalConfig
     *            - EvalConfig to find
     * @param sourceType
     *            - Local/HDFS
     * @return - path of evaluation set performance
     */
    public String getEvalPerformancePath(EvalConfig evalConfig, SourceType sourceType) {
        String evalPerformancePath = getPreferPath(evalConfig.getCustomPaths(), Constants.KEY_PERFORMANCE_PATH);
        if(StringUtils.isBlank(evalPerformancePath)) {
            return getEvalFilePath(evalConfig.getName(), Constants.EVAL_PERFORMANCE, sourceType);
        } else {
            return new Path(evalPerformancePath, Constants.EVAL_PERFORMANCE).toString();
        }
    }

    public String getEvalMatrixPath(EvalConfig evalConfig, SourceType sourceType) {
        String evalMatrixPath = getPreferPath(evalConfig.getCustomPaths(), Constants.KEY_CONFUSION_MATRIX_PATH);
        if(StringUtils.isBlank(evalMatrixPath)) {
            return getEvalFilePath(evalConfig.getName(), Constants.EVAL_MATRIX, sourceType);
        } else {
            return new Path(evalMatrixPath, Constants.EVAL_MATRIX).toString();
        }
    }

    /**
     * Get the file path for specified name under evaluation set home directory
     * 
     * @param evalName
     *            - evalset name to find
     * @param specifiedFileName
     *            - the specified file name
     * @param sourceType
     *            - Local/HDFS
     * @return path of specified file
     */
    public String getEvalFilePath(String evalName, String specifiedFileName, SourceType sourceType) {
        return (new Path(this.getEvalSetPath(evalName, sourceType), specifiedFileName)).toString();
    }

    public String getEvalLocalMultiMatrixFile(String evalName) {
        return (new Path(this.getEvalSetPath(evalName, SourceType.LOCAL), "multi-eval-confustion-matrix.csv"))
                .toString();
    }

    public String getModelSetPath(SourceType sourceType) {
        switch(sourceType) {
            case LOCAL:
                return this.getModelSetLocalPath().toString();
            case HDFS:
                return this.getModelSetHdfsPath().toString();
            default:
                // Others, maybe be we will support S3 in future
                throw new NotImplementedException("Source type - " + sourceType.name() + " is not supported yet!");
        }
    }

    /**
     * Get the local home directory for Model
     * 
     * @return - the Path of local home directory
     */
    private Path getModelSetLocalPath() {
        return (otherConfigs != null && otherConfigs.get(Constants.SHIFU_CURRENT_WORKING_DIR) != null) ? new Path(
                otherConfigs.get(Constants.SHIFU_CURRENT_WORKING_DIR).toString()) : new Path(".");
    }

    /**
     * Get the HDFS home directory for Model
     * 
     * @return - the Path of HDFS home directory
     */
    private Path getModelSetHdfsPath() {
        String modelSetPath = this.getPreferPath(modelConfig.getBasic().getCustomPaths(),
                Constants.KEY_HDFS_MODEL_SET_PATH);
        return (StringUtils.isBlank(modelSetPath) ? new Path(Constants.MODEL_SETS, modelConfig.getBasic().getName())
                : new Path(modelSetPath, modelConfig.getBasic().getName()));
    }

    /**
     * Get the relative path to model home directory by source type
     * 
     * @param path
     *            - the path to find
     * @param sourceType
     *            - Local/HDFS
     * @return the relative path to the model home directory
     */
    public String getPathBySourceType(Path path, SourceType sourceType) {
        switch(sourceType) {
            case LOCAL:
                return new Path(getModelSetLocalPath(), path).toString();
            case HDFS:
                return ShifuFileUtils.getFileSystemBySourceType(sourceType)
                        .makeQualified(new Path(getModelSetHdfsPath(), path)).toString();
            default:
                // Others, maybe be we will support S3 in future
                throw new NotImplementedException("Source type - " + sourceType.name() + " is not supported yet!");
        }
    }

    /**
     * Get the relative path to model home directory by source type
     * 
     * @param path
     *            - the path to find
     * @param sourceType
     *            - Local/HDFS
     * @return the relative path to the model home directory
     */
    public String getPathBySourceType(String path, SourceType sourceType) {
        return getPathBySourceType(new Path(path), sourceType);
    }

    /**
     * Return local correlation csv path
     * 
     * @return the local correlation csv path
     */
    public String getLocalCorrelationCsvPath() {
        return getPathBySourceType(CORRELATION_CSV, SourceType.LOCAL);
    }

    /**
     * Get the prefer path for files. Prefer path means:
     * - if the user set the path in customPaths, try to use it
     * - or return null
     * 
     * @param customPaths
     *            - map of customer setting path
     * @param key
     *            - path key in ModelConfig
     * @return - path for files
     */
    private String getPreferPath(Map<String, String> customPaths, String key) {
        if(customPaths == null || customPaths.size() == 0) {
            return null;
        }

        return customPaths.get(key);
    }

    public String getLocalFeatureImportancePath() {
        return new Path(getLocalFeatureImportanceFolder(), FEATURE_IMPORTANCE_FILE).toString();
    }

    public String getLocalFeatureImportanceFolder() {
        return getPathBySourceType(new Path("featureImportance"), SourceType.LOCAL);
    }

    /**
     * @return the otherConfigs
     */
    public Map<String, Object> getOtherConfigs() {
        return otherConfigs;
    }

    /**
     * Get the train data path for assemble model
     * 
     * @return - train data path for assemble model
     */
    public String getSubModelsAssembleTrainData() {
        return getPathBySourceType(new Path(Constants.TMP, "AssembleTrainData"), this.modelConfig.getDataSet()
                .getSource());
    }

    /**
     * Get the eval data path for assemble model
     * 
     * @param evalName
     *            - evalset name
     * @param sourceType
     *            - Local/HDFS
     * @return - eval data path for assemble model
     */
    public String getSubModelsAssembleEvalData(String evalName, SourceType sourceType) {
        return getPathBySourceType(new Path(Constants.TMP, evalName + "AssembleEvalData"), sourceType);
    }

    /**
     * Get the shuffle data path
     * @return - the shuffle data path
     */
    public String getShuffleDataPath() {
        return getShuffleDataPath(modelConfig.getDataSet().getSource());
    }

    /**
     * Get the shuffle data path according SourceType
     * @param sourceType - Local/HDFS
     * @return - the shuffle data path
     */
    private String getShuffleDataPath(SourceType sourceType) {
        return getPathBySourceType(new Path(Constants.TMP, Constants.SHUFFLED_DATA_PATH), sourceType);
    }

    /**
     * Get the backup ColumnConfig
     * @return - the ColumnConfig.json path for backup
     */
    public String getBackupColumnConfig() {
        return getPathBySourceType(new Path(Constants.TMP, Constants.COLUMN_CONFIG_JSON_FILE_NAME), SourceType.LOCAL);
    }


    /**
     * Get the varsel auto filter history to let user have the opportunity to change
     * @return - the varsel.history path for variable auto filter
     */
    public String getVarSelHistory() {
        return getPathBySourceType(new Path(Constants.TMP, Constants.VAR_SEL_HISTORY), SourceType.LOCAL);
    }

    /**
     * Get the correlation export path
     * @return - the correlation path for export
     */
    public String getCorrExportPath() {
        return getPathBySourceType(new Path(Constants.TMP, Constants.CORR_EXPORT_PATH), SourceType.LOCAL);
    }
}
