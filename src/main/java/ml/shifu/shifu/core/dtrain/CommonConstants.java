/*
 * Copyright [2012-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain;

public interface CommonConstants {

    public static final double DEFAULT_SIGNIFICANCE_VALUE = 1.0d;
    public static final String MAPREDUCE_PARAM_FORMAT = "-D%s=%s";
    public static final String GUAGUA_OUTPUT = "guagua.output";

    public static final String LR_REGULARIZED_CONSTANT = "RegularizedConstant";

    public static final String LR_LEARNING_RATE = "LearningRate";

    public static final String REG_LEVEL_KEY = "L1orL2";

    public static final String MODELSET_SOURCE_TYPE = "shifu.source.type";

    public static final String SHIFU_COLUMN_CONFIG = "shifu.column.config";

    public static final String SHIFU_MODEL_CONFIG = "shifu.model.config";

    public static final String SHIFU_TMP_MODELS_FOLDER = "shifu.tmp.models.folder";

    public static final String SHIFU_DRY_DTRAIN = "shifu.dry.dtrain";

    public static final String SHIFU_TRAINER_ID = "shifu.trainer.id";

    public static final String SHIFU_DTRAIN_PROGRESS_FILE = "shifu.progress.file";

    public static final String DEFAULT_COLUMN_SEPARATOR = "|";

    public static final String RF_ALG_NAME = "RF";

    public static final String GBT_ALG_NAME = "GBT";

    public static final String GS_VALIDATION_ERROR = "gridsearch.validation.error";

    public static final String CROSS_VALIDATION_DIR = "shifu.crossValidation.dir";

    public static final String SHIFU_TRAIN_BAGGING_INPARALLEL = "shifu.train.bagging.inparallel";

    public static final String CONTINUOUS_TRAINING = "shifu.continuous.training";

    public static final String SHIFU_DT_MASTER_CHECKPOINT_INTERVAL = "shifu.dt.master.checkpoint.interval";

    public static final String SHIFU_DT_MASTER_CHECKPOINT_FOLDER = "shifu.dt.master.checkpoint.folder";

    /**
     * Version 2: support final selected columns
     * Version 3: in Node to change wgtCnt float to double
     * Version 4: change trees in IndependentTreeModel to support bagging of RF and GBDT
     */
    public static final int TREE_FORMAT_VERSION = 4;

    public static final int NN_FORMAT_VERSION = 1;

    public static final String NUM_HIDDEN_LAYERS = "NumHiddenLayers";

    public static final String ACTIVATION_FUNC = "ActivationFunc";

    public static final String NUM_HIDDEN_NODES = "NumHiddenNodes";

    public static final String LEARNING_RATE = "LearningRate";

    public static final String DROPOUT_RATE = "DropoutRate";

    public static final String PROPAGATION = "Propagation";

    public static final String SHIFU_NN_FEATURE_SUBSET = "shifu.nn.feature.subset";

    public static final String SHIFU_TREE_CHECKPOINT_INTERVAL = "shifu.tree.checkpoint.interval";

    public static final String NAMESPACE_DELIMITER = "::";

    public static final long NOT_CONFIGURED_BAGGING_SEED = -1L;

    public static final long MAX_RECORDS_PER_WORKER = 50000L;
    
    public static final int PART_FILE_COUNT_THRESHOLD = 50;

}
