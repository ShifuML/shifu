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
package ml.shifu.shifu.exception;


/**
 * Shifu error code
 */
public enum ShifuErrorCode {
    /**
     * Configuration Error 400 ~ 500
     */
    ERROR_SHIFU_CONFIG(400, "Errors happen when loading shifuconfig"),ERROR_GRID_SEARCH_FILE_CONFIG(501,
            "Errors happen when loading grid search file config"),

    /**
     * Configuration Error 500 ~ 600
     */
    ERROR_MODELCONFIG_VALIDATION(500, "Errors happen when validating ModelConfig.json"),

    /**
     * AKKA Error 601 ~ 700
     */
    ERROR_AKKA_EXECUTE_EXCEPTION(601, "Exception happenen, when AKKA executing"),

    /*
     * File/System error: 1001 - 1050
     */
    ERROR_INPUT_NOT_FOUND(1001, "The input data is not found"), ERROR_HEADER_NOT_FOUND(1002,
            "The pig_header is not found"), ERROR_LOAD_MODELCONFIG(1003, "Could not load ModelConfig"), ERROR_WRITE_MODELCONFIG(
            1004, "Could not write ModelConfig file"), ERROR_LOAD_COLCONFIG(1005, "Could not load ColumnConfig"), ERROR_WRITE_COLCONFIG(
            1006, "Could not write ColumnConfig file"), ERROR_GET_HDFS_SYSTEM(1007,
            "Could not initialize the hdfs system"), ERROR_GET_LOCAL_SYSTEM(1008,
            "Could not initialize the local file system"), ERROR_CLOSE_READER(1009, "Could not close the reader"), ERROR_DETELE_LOCAL_FILE(
            1010, "Could not delete local file, please manually delete it "), ERROR_DELETE_HDFS_FILE(1011,
            "Could not delete hdfs file, please manually delete it"), ERROR_RUNNING_PIG_JOB(1012,
            "Could not running the pig job or pig job occur internal error, please check your pig log"), ERROR_COPY_TO_HDFS(
            1013, "Could not copy file to hdfs"), ERROR_COPY_DATA(1014,
            "Could not copy data, it could be the source data unavailable or targe path is lock by system"), ERROR_NO_EVAL_SET(
            1015, "Could not copy eval file to hdfs"),
    /*
     * ModelConfig validated 1051 - 1100
     */
    ERROR_MODELCONFIG_NOT_VALIDATION(1051, "The ModelConfig file did not pass the validation."), ERROR_UNSUPPORT_ALG(
            1052, "Un-support algorithm, make sure your ModelConfig.json -> algorithm is NN/SVM/LR/DT/RF/GBT"), ERROR_UNSUPPORT_MODE(
            1053, "Un-support mode, make sure your ModelConfig.json -> mode is \"local\" or \"mapred\"."), ERROR_UNSUPPORT_RUNMODE(
            1054, "Un-support running mode, make sure your ModelConfig.json -> runMode is pig or akka"), ERROR_GRIDCONFIG_NOT_VALIDATION(
            1055, "The grid search config file did not pass the validation."),

    /*
     * ColumnConfig validated 1101 - 1150
     */

    /*
     * data validate 1151 - 1200
     */
    ERROR_EXCEED_COL(1151, "The input data length is more than column config"), ERROR_LESS_COL(1152,
            "The input data length is less than column config"), ERROR_NO_EQUAL_COLCONFIG(1153,
            "The input data length is not equal to column config size"), ERROR_NO_TARGET_COLUMN(1154,
            "There is no target column in training data"), ERROR_INVALID_TARGET_VALUE(1155,
            "Invalid target value,target value must be 1 or 0"),

    /*
     * model validate 1201 - 1250
     */
    ERROR_MODEL_FILE_NOT_FOUND(1250, "The model file is not found!"), ERROR_FAIL_TO_LOAD_MODEL_FILE(1251,
            "Fail to load the model file"),

    /*
     * model eval 1301 - 1350
     */
    ERROR_MODEL_EVALSET_DOESNT_EXIST(1301, "The evalset doesn't exist!"), ERROR_MODEL_EVALSET_ALREADY_EXIST(1302,
            "The evalset already exists!"), ERROR_EVALSCORE(1303, "the evalscore file is empty"), ERROR_EVALCONFMTR(
            1304, "the confusion matrix file is empty"), ERROR_EVAL_SELECTOR_EMPTY(1305,
            "the performanceScoreSelector is empty or not setting properly"), ERROR_EVAL_NO_EVALSCORE_HEADER(1306,
            "there is no header for EvalScore"), ERROR_EVAL_TARGET_NOT_FOUND(1307,
            "target column is not found in the header of EvalScore"),

    /**
     * Exception in d-training client
     */
    ERROR_MODEL_D_TRAIN_CLIENT_EXCEPTION(1401, "Exception in d-training client.");

    /**
     * code
     */
    private final int code;

    /**
     * description
     */
    private final String description;

    /**
     * Constructor, not public
     * 
     * @param code
     *            the code
     * @param description
     *            the description
     */
    private ShifuErrorCode(int code, String description) {
        this.code = code;
        this.description = description;
    }

    /**
     * description getter
     * 
     * @return description
     */
    public String getDescription() {
        return description;
    }

    /**
     * code getter
     * 
     * @return code
     */
    public int getCode() {
        return code;
    }

    /**
     * user to string
     */
    @Override
    public String toString() {
        return code + ": " + description;
    }

}
