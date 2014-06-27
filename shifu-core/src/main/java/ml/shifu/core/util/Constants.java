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
package ml.shifu.core.util;

/**
 * Global constants class
 */
public interface Constants {

    public static final String version = "0.2.0";

    public static final String MODELS = "models";

    public static final String COLUMN_CONFIG_JSON_FILE_NAME = "ColumnConfig.json";

    public static final String MODEL_CONFIG_JSON_FILE_NAME = "ModelConfig.json";

    public static final String MODEL_SETS = "ModelSets";

    public static final String TMP = "tmp";

    public static final String PIG_HEADER = ".pig_header";

    public static final String REASON_CODE_MAP_JSON = "ReasonCodeMap.json";

    public static final String DEFAULT_JOB_QUEUE = "default";

    public static final String JOB_QUEUE = "queue_name";

    public static final String PATH_TRAIN_SCORE = "pathTrainScore";
    public static final String PATH_BIN_AVG_SCORE = "pathBinAvgScore";
    public static final String PATH_SELECTED_RAW_DATA = "pathSelectedRawData";
    public static final String PATH_COLUMN_CONFIG = "path_column_config";
    public static final String PATH_MODEL_CONFIG = "path_model_config";
    public static final String PATH_PRE_TRAINING_STATS = "path_pre_training_stats";
    public static final String PATH_NORMALIZED_DATA = "pathNormalizedData";
    public static final String PATH_RAW_DATA = "path_raw_data";
    public static final String PATH_JAR = "path_jar";

    public static final String STATS_SAMPLE_RATE = "statsSampleRate";
    public static final String WITH_SCORE = "with_score";
    public static final String SOURCE_TYPE = "source_type";
    public static final String NUM_PARALLEL = "num_parallel";

    public static final String DERIVED = "derived_";

    public static final int DEFAULT_IDEAL_VALUE = -1;

    public static final double DEFAULT_CUT_OFF = 6.0;

    public static final String LR = "lr";
    public static final String SVM = "svm";
    public static final String NN = "nn";

    public static final String BZ2_SUFFIX = ".bz2";
    public static final String GZ_SUFFIX = ".gz";

    public static final String MAPREDUCE_OUTPUT_PREFIX = "part-";
    public static final String HIDDEN_FILES = ".";
    public static final String BACK_SLASH = "\\";
    public static final String SLASH = "/";
    public static final String REGEX_MULTIPLE_INPUTS = "[{}]";
    public static final String COMMA = ",";
    public static final String PIG_COLUMN_SEPARATOR = "::";

    public static final String LOCAL_MODEL_CONFIG_JSON = "./ModelConfig.json";
    public static final String LOCAL_COLUMN_CONFIG_JSON = "./ColumnConfig.json";

    public static final double TOLERANCE = 0.00001d;

    public static final String DEFAULT_MODELS_TMP_FOLDER = "modelsTmp";
    public static final String BACKUPNAME = "backup_models";

    public static final String CONTACT_MESSAGE = "Error! Please check the log file for more information.";

    public static final String EVAL_DIR = "evals";
    public static final String EVAL_NORMALIZED = "EvalNormalized";
    public static final String EVAL_SCORE = "EvalScore";
    public static final String EVAL_PERFORMANCE = "EvalPerformance.json";
    public static final String EVAL_MATRIX = "EvalConfusionMatrix";

    public static final String KEY_HDFS_MODEL_SET_PATH = "hdfsModelSetPath";
    public static final String KEY_MODELS_PATH = "modelsPath";
    public static final String KEY_SCORE_PATH = "scorePath";
    public static final String KEY_PERFORMANCE_PATH = "performancePath";
    public static final String KEY_CONFUSION_MATRIX_PATH = "confusionMatrixPath";

    public static final String PRE_TRAINING_STATS = "PreTrainingStats";
    public static final String SELECTED_RAW_DATA = "SelectedRawData";
    public static final String NORMALIZED_DATA = "NormalizedData";
    public static final String TRAIN_SCORES = "TrainScores";
    public static final String BIN_AVG_SCORE = "BinAvgScore";

    public static final String KEY_PRE_TRAIN_STATS_PATH = "preTrainStatsPath";
    public static final String KEY_SELECTED_RAW_DATA_PATH = "selectedRawDataPath";
    public static final String KEY_NORMALIZED_DATA_PATH = "normalizedDataPath";
    public static final String KEY_TRAIN_SCORES_PATH = "trainScoresPath";
    public static final String KEY_BIN_AVG_SCORE_PATH = "binAvgScorePath";

    public static final String DEFAULT_META_COLUMN_FILE = "meta.column.names";
    public static final String DEFAULT_CATEGORICAL_COLUMN_FILE = "categorical.column.names";
    public static final String DEFAULT_FORCESELECT_COLUMN_FILE = "forceselect.column.names";
    public static final String DEFAULT_FORCEREMOVE_COLUMN_FILE = "forceremove.column.names";
    public static final String DEFAULT_EVALSCORE_META_COLUMN_FILE = "score.meta.column.names";
}
