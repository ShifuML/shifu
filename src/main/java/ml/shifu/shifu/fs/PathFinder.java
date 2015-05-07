/**
 * Copyright [2012-2014] eBay Software Foundation
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
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;

import java.util.Map;

/**
 * <p/>
 * {@link PathFinder} is used to obtain all files which can be used in our framework. Some are used for training,
 * evaling, performance ...
 * <p/>
 * <p/>
 * {@link #modelConfig} should be passed as parameter in constructor
 */
public class PathFinder {

    private static final String REASON_CODE_PATH = "common/ReasonCodeMapV3.json";
    private static final String SHIFU_JAR_PATH = "lib/*.jar";

    /**
     * {@link PathFinder#modelConfig} is used to get global setting for model config path.
     */
    private ModelConfig modelConfig;

    /**
     * Constructor with valid parameter modelConfig
     * 
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
     * Get absolute path with SHIFU_HOME env.
     * - if the path is already absolute path, just return
     * - or assume it is relative path to SHIFU_HOME
     * 
     * @param path
     * @return absolute path
     */
    public String getAbsolutePath(String path) {
        return (new Path(path)).isAbsolute() ? path : new Path(Environment.getProperty(Environment.SHIFU_HOME), path)
                .toString();
    }

    /**
     * Get project jar file path name.
     * - Since the Jar Path is only used in pig code compiling, just return local path
     * 
     * @return path of SHIFU dependent jars
     */
    public String getJarPath() {
        return getAbsolutePath(SHIFU_JAR_PATH);
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
            return getPathBySourceType(new Path(Constants.TMP, Constants.KEY_AUTO_TYPE_PATH), sourceType);
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

    /**
     * @param sourceType
     * @return
     */
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
     * @return path of evaluation set home directory
     */
    public String getEvalSetPath(EvalConfig evalConfig) {
        return getEvalSetPath(evalConfig, evalConfig.getDataSet().getSource());
    }

    /**
     * Get evaluation set home directory, something like <Model>/eval/<evalName>
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
     * Get evaluation set home directory, something like <Model>/eval/<evalName>
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
     * Get the path of evaluation set performance
     * 
     * @param evalConfig
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

    /**
     * Get path for evaluation matrix
     * 
     * @param evalConfig
     * @param sourceType
     * @return
     */
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

    /**
     * @param hdfs
     * @return
     */
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
        return new Path(".");
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

}