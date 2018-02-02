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
package ml.shifu.shifu.util;

/**
 * Global constants class
 */
public interface Constants {

    public static final String version = "0.11.0";

    public static final String MODELS = "models";

    public static final String VarSels = "varsels";

    public static final String COLUMN_CONFIG_JSON_FILE_NAME = "ColumnConfig.json";

    public static final String MODEL_CONFIG_JSON_FILE_NAME = "ModelConfig.json";

    public static final String COMBO_CONFIG_JSON_FILE_NAME = "ComboTrain.json";

    public static final String COLUMN_STATS_CSV_FILE_NAME = "ColumnStats.csv";

    public static final String MODEL_SETS = "ModelSets";

    public static final String TMP = "tmp";
    public static final String MODELS_TMP = "modelsTmp";

    public static final String PIG_HEADER = ".pig_header";

    public static final String REASON_CODE_MAP_JSON = "ReasonCodeMap.json";

    public static final String DEFAULT_JOB_QUEUE = "default";

    public static final int DEFAULT_MAPRED_TIME_OUT = 600000;

    public static final String JOB_QUEUE = "queue_name";

    public static final String PATH_TRAIN_SCORE = "pathTrainScore";
    public static final String PATH_BIN_AVG_SCORE = "pathBinAvgScore";
    public static final String PATH_SELECTED_RAW_DATA = "pathSelectedRawData";
    public static final String PATH_COLUMN_CONFIG = "path_column_config";
    public static final String PATH_MODEL_CONFIG = "path_model_config";
    public static final String PATH_PRE_TRAINING_STATS = "path_pre_training_stats";
    public static final String PATH_STATS_BINNING_INFO = "path_stats_binning_info";
    public static final String PATH_STATS_PSI_INFO = "path_psi";

    public static final String PATH_NORMALIZED_DATA = "pathNormalizedData";
    public static final String PATH_RAW_DATA = "path_raw_data";
    public static final String PATH_JAR = "path_jar";
    public static final String IS_COMPRESS = "is_compress";
    public static final String IS_NORM_FOR_CLEAN = "is_norm_for_clean";

    public static final String STATS_SAMPLE_RATE = "statsSampleRate";
    public static final String WITH_SCORE = "with_score";
    public static final String SOURCE_TYPE = "source_type";
    public static final String NUM_PARALLEL = "num_parallel";
    public static final String DATASET_NAME = "data_set";

    public static final String DERIVED = "derived_";

    public static final int DEFAULT_IDEAL_VALUE = -1;

    public static final double DEFAULT_CUT_OFF = 6.0;

    public static final String LR = "lr";
    public static final String SVM = "svm";
    public static final String NN = "nn";
    public static final String GBT = "GBT";

    public static final String BZ2_SUFFIX = ".bz2";
    public static final String GZ_SUFFIX = ".gz";

    public static final String MAPREDUCE_OUTPUT_PREFIX = "part-";
    public static final String HIDDEN_FILES = ".";
    public static final String BACK_SLASH = "\\";
    public static final String SLASH = "/";
    public static final String REGEX_MULTIPLE_INPUTS = "[{}]";
    public static final String COMMA = ",";
    public static final String PIG_COLUMN_SEPARATOR = "::";
    public static final String PIG_FULL_COLUMN_SEPARATOR = "_";

    public static final String LOCAL_MODEL_CONFIG_JSON = "./ModelConfig.json";
    public static final String LOCAL_COLUMN_CONFIG_JSON = "./ColumnConfig.json";

    public static final double TOLERANCE = 0.00001d;

    public static final String DEFAULT_MODELS_TMP_FOLDER = "modelsTmp";
    public static final String BACKUPNAME = "backup_models";

    public static final String CONTACT_MESSAGE = "Error! Please check the log file for more information.";

    public static final String EVAL_DIR = "evals";
    public static final String EVAL_NORMALIZED = "EvalNormalized";
    public static final String EVAL_SCORE = "EvalScore";
    public static final String EVAL_META_SCORE = "EvalMetaScore";
    public static final String EVAL_PERFORMANCE = "EvalPerformance.json";
    public static final String EVAL_MATRIX = "EvalConfusionMatrix";

    public static final String KEY_HDFS_MODEL_SET_PATH = "hdfsModelSetPath";
    public static final String KEY_MODELS_PATH = "modelsPath";
    public static final String KEY_SCORE_PATH = "scorePath";
    public static final String KEY_PERFORMANCE_PATH = "performancePath";
    public static final String KEY_CONFUSION_MATRIX_PATH = "confusionMatrixPath";

    public static final String COMBO_EVAL_TRAIN = "EvalTrain";
    public static final String COMBO_ASSEMBLE = "assemble";

    public static final String BIN_BOUNDRY_DELIMITER = "\u0001";

    public static final String DEFAULT_ESCAPE_DELIMITER = "\\|";

    public static final String DEFAULT_DELIMITER = "|";

    public static final String BINNING_INFO_FILE_NAME = "binning_info.txt";

    public static final String AUTO_TYPE_PATH = "AutoTypePath";
    public static final String PRE_TRAINING_STATS = "PreTrainingStats";
    public static final String STATS_SMALL_BINS = "StatsSmallBins";
    public static final String SELECTED_RAW_DATA = "SelectedRawData";
    public static final String NORMALIZED_DATA = "NormalizedData";
    public static final String CLEANED_DATA = "CleanedData";
    public static final String NORMALIZED_VALIDATION_DATA = "NormalizedValidationData";
    public static final String CLEANED_VALIDATION_DATA = "CleanedValidationData";
    public static final String SHUFFLED_DATA_PATH = "ShuffledData";

    public static final String VAR_SEL_HISTORY = "varsel.history";
    public static final String CORR_EXPORT_PATH = "vars_corr.csv";
    public static final String TRAIN_SCORES = "TrainScores";
    public static final String BIN_AVG_SCORE = "BinAvgScore";
    public static final String CORRELATION_PATH = "CorrelationPath";
    public static final String TAB_STR = "\t";

    public static final String KEY_POSTTRAIN_OUT_PATH = "posttrainOutputPath";
    public static final String KEY_PRE_TRAIN_STATS_PATH = "preTrainStatsPath";
    public static final String KEY_PRE_PSI_PATH = "StatsPSIPath";
    public static final String KEY_SELECTED_RAW_DATA_PATH = "selectedRawDataPath";
    public static final String KEY_NORMALIZED_DATA_PATH = "normalizedDataPath";
    public static final String KEY_CLEANED_DATA_PATH = "cleanedDataPath";
    public static final String KEY_NORMALIZED_VALIDATION_DATA_PATH = "normalizedValidationDataPath";
    public static final String KEY_CLEANED_VALIDATION_DATA_PATH = "cleanedValidationDataPath";

    public static final String KEY_VARSLECT_STATS_PATH = "varSelectStatsPath";
    public static final String KEY_TRAIN_SCORES_PATH = "trainScoresPath";
    public static final String KEY_BIN_AVG_SCORE_PATH = "binAvgScorePath";
    public static final String KEY_AUTO_TYPE_PATH = "autoTypePath";
    public static final String KEY_CORRELATION_PATH = "correlationPath";

    public static final String DEFAULT_META_COLUMN_FILE = "meta.column.names";
    public static final String DEFAULT_CATEGORICAL_COLUMN_FILE = "categorical.column.names";
    public static final String DEFAULT_HYBRID_COLUMN_FILE = "hybrid.column.names";
    public static final String DEFAULT_CANDIDATE_COLUMN_FILE = "candidate.column.names";
    public static final String DEFAULT_FORCESELECT_COLUMN_FILE = "forceselect.column.names";
    public static final String DEFAULT_FORCEREMOVE_COLUMN_FILE = "forceremove.column.names";
    public static final String DEFAULT_EVALSCORE_META_COLUMN_FILE = "score.meta.column.names";
    public static final String DEFAULT_EXPRESSION_COLUMN_FILE = "filter.expressions";

    public static final String VAR_SEL_MASTER_CONDUCTOR = "dvarsel.master.conductor.cls";
    public static final String VAR_SEL_WORKER_CONDUCTOR = "dvarsel.worker.conductor.cls";
    public static final String VAR_SEL_COLUMN_IDS_OUPUT = "dvarsle.column.ids.output";

    public static final String SHIFU_COLUMN_CONFIG = "shifu.column.config";

    public static final String SHIFU_MODEL_CONFIG = "shifu.model.config";

    public static final String SHIFU_MODELSET_SOURCE_TYPE = "shifu.modelset.source.type";

    public static final String SHIFU_VARSELECT_FILTEROUT_RATIO = "shifu.varselect.filterout.ratio";

    public static final String SHIFU_VARSELECT_FILTER_NUM = "shifu.varselect.filter.num";

    public static final String SHIFU_VARSELECT_FILTEROUT_TYPE = "shifu.varselect.filterout.type";

    public static final String SHIFU_DEFAULT_VARSEL_SE_MULTI = "true";

    public static final String SHIFU_VARSEL_SE_MULTI = "shifu.varsel.se.multi";

    public static final String SHIFU_VARSEL_SE_MULTI_THREAD = "shifu.varsel.se.multi.thread";

    public static final int SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD = 6;

    public static final String FILTER_BY_ST = "ST";

    public static final String FILTER_BY_SE = "SE";

    public static final String FILTER_BY_VOTED = "V";

    public static final String FILTER_BY_FI = "FI";

    public static final String FILTER_BY_KS = "KS";

    public static final String FILTER_BY_IV = "IV";

    public static final String FILTER_BY_MIX = "MIX";

    public static final String FILTER_BY_PARETO = "PARETO";

    public static final float SHIFU_DEFAULT_VARSELECT_FILTEROUT_RATIO = 0.05f;

    public static final int SHIFU_DEFAULT_VARSELECT_FILTER_NUM = -1;

    public static final String COUNTER_WNEGTAGS = "WNEGTAGS";

    public static final String COUNTER_WPOSTAGS = "WPOSTAGS";

    public static final String COUNTER_POSTAGS = "POSTAGS";

    public static final String COUNTER_NEGTAGS = "NEGTAGS";

    public static final String SHIFU_GROUP_COUNTER = "SHIFU_COUNTER";

    public static final long EVAL_COUNTER_WEIGHT_SCALE = 1000l;

    public static final String COUNTER_RECORDS = "RECORDS";

    public static final String TOTAL_MODEL_RUNTIME = "TOTAL_MODEL_RUNTIME";

    public static final String COUNTER_MAX_SCORE = "MAX_SCORE";

    public static final String COUNTER_MIN_SCORE = "MIN_SCORE";

    public static final String SHIFU_VARSELECT_SE_OUTPUT_NAME = "se";

    public static final String DEFAULT_CHARSET = "UTF-8";

    public static final String SHIFU_STATS_EXLCUDE_MISSING = "shifu.stats.exlcudeMissing";

    public static final String COLUMN_META_FOLDER_NAME = "columns";

    public static final String POST_TRAIN_OUTPUT_SCORE = "score";

    public static final String SHIFU_EVAL_MAXMIN_SCORE_OUTPUT = "shifu.eval.maxmin.score.output";

    public static final String SHIFU_DTRAIN_PARALLEL = "shifu.dtrain.parallel";

    public static final String SHIFU_TMPMODEL_COPYTOLOCAL = "shifu.tmpmodel.copytolocal";

    public static final String SHIFU_NORM_SHUFFLE_SIZE = "shifu.norm.shuffle.size";

    public static final String SHIFU_NORM_PREFER_PART_SIZE = "shifu.norm.prefer.part.size";

    public static final String SHIFU_SCORE_SCALE = "shifu.score.scale";

    public static final String SHIFU_CORRELATION_MULTI_THREADS = "shifu.correlation.multi.threads";

    public static final String SHIFU_CORRELATION_MULTI = "shifu.correlation.multi";

    public static final String SHIFU_CURRENT_WORKING_DIR = "shifu.current.working.dir";

    public static final String SHIFU_CORRELATION_COMPUTE_ALL = "shifu.correlation.computeAll";

    public static final String CATEGORICAL_GROUP_VAL_DELIMITER = "@^";

    public static final String SHIFU_NAMESPACE_STRICT_MODE = "shifu.namespace.strict.mode";

    public static final String EMPTY_CATEGORY = "";

    /**
     * The limitation of max categorical value length
     */
    public static final int MAX_CATEGORICAL_VAL_LEN = 10 * 1024;

    /**
     * Experience value from modeler
     */
    public static final int MAX_CATEGORICAL_BINC_COUNT = 10000;

    public static final String SHIFU_MAX_CATEGORY_SIZE = "shifu.max.category.size";

    public static final String SHIFU_NN_INDEPENDENT_MODEL = "shifu.nn.independent.model";

    public static final String SHIFU_NN_BINARY_MODEL_PATH = "shifu.nn.binary.model.path";

    public static final String HYBRID_BIN_STR_DILIMETER = ";;;";

    public static final String SHIFU_EVAL_PARALLEL_NUM = "shifu.eval.parallel.num";

    public static final String SHIFU_EVAL_PARALLEL = "shifu.eval.parallel";

    public static final String SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER = "@@";

    public static final String SHIFU_STATS_FILTER_EXPRESSIONS = "shifu.stats.filter.expressions";

    public static final String IS_COMPUTE_PSI = "IS_COMPUTE_PSI";
    public static final String IS_COMPUTE_CORR = "IS_COMPUTE_CORR";
    public static final String IS_REBIN = "IS_RE_BIN";

    public static final String REQUEST_VARS = "REQUEST_VARS";
    public static final String EXPECTED_BIN_NUM = "EXPECTED_BIN_NUM";
    public static final String IV_KEEP_RATIO = "IV_KEEP_RATIO";
    public static final String MINIMUM_BIN_INST_CNT = "MINIMUM_BIN_INST_CNT";

    public static final String IS_TO_RESET = "IS_TO_RESET";
    public static final String IS_TO_LIST = "IS_TO_LIST";
    public static final String IS_TO_FILTER_AUTO = "IS_TO_FILTER_AUTO";
    public static final String IS_TO_RECOVER_AUTO = "IS_TO_RECOVER_AUTO";

    /**
     * GBT score range is not in [0, 1], to make it in [0, 1], such strategies are provided with case insensitive.
     */
    public static String GBT_SCORE_RAW_CONVETER = "RAW";
    public static String GBT_SCORE_SIGMOID_CONVETER = "SIGMOID";
    public static String GBT_SCORE_OLD_SIGMOID_CONVETER = "OLD_SIGMOID";
    public static String GBT_SCORE_CUTOFF_CONVETER = "CUTOFF";
    public static String GBT_SCORE_HALF_CUTOFF_CONVETER = "HALF_CUTOFF";
    public static String GBT_SCORE_MAXMIN_SCALE_CONVETER = "MAXMIN";

    public static final String SHIFU_SEGMENT_EXPRESSIONS = "shifu.segment.expressions";
    public static final String SHIFU_NORM_CATEGORY_MISSING_NORM = "shifu.norm.category.missing.norm";

}
