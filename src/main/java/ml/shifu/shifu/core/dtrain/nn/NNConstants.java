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
package ml.shifu.shifu.core.dtrain.nn;

/**
 * Constants in guagua mapreduce.
 */
public class NNConstants {

    public static final String GUAGUA_NN_LEARNING_RATE = "guagua.nn.learning.rate";
    public static final String GUAGUA_NN_THREAD_COUNT = "guagua.nn.thread.count";
    public static final String GUAGUA_NN_ALGORITHM = "guagua.nn.algorithm";
    public static final String GUAGUA_NN_OUTPUT_NODES = "guagua.nn.output.nodes";
    public static final String GUAGUA_NN_HIDDEN_NODES = "guagua.nn.hidden.nodes";
    public static final String GUAGUA_NN_INPUT_NODES = "guagua.nn.input.nodes";

    public static final String GUAGUA_NN_DEFAULT_LEARNING_RATE = "0.1";
    public static final int GUAGUA_NN_DEFAULT_THREAD_COUNT = 0;
    public static final String GUAGUA_NN_DEFAULT_ALGORITHM = "Q";
    public static final int GUAGUA_NN_DEFAULT_OUTPUT_NODES = 1;
    public static final int GUAGUA_NN_DEFAULT_HIDDEN_NODES = 2;
    public static final int GUAGUA_NN_DEFAULT_INPUT_NODES = 100;
    public static final String NN_RECORD_SCALE = "nn.record.scale";
    public static final String NN_TEST_SCALE = "nn.test.scale";

    public static final String MAPRED_JOB_QUEUE_NAME = "mapred.job.queue.name";

    public static final String MAPRED_TASK_TIMEOUT = "mapred.task.timeout";


    public static final String TESTING_EGB = "testing.egb";

    public static final String TRAINING_EGB = "training.egb";

    public static final String NN_SIN = "sin";
    public static final String NN_LOG = "log";
    public static final String NN_TANH = "tanh";
    public static final String NN_SIGMOID = "sigmoid";
    public static final String NN_LINEAR = "linear";
    public static final String NN_RELU = "relu";

    public static final String LIB_JAR_SEPARATOR = ",";

    public static final String LIB_PATH_NAME = "lib";

    public static final String NN_ALG_NAME = "NN";

    public static final String JAVA_IO_TMPDIR = "java.io.tmpdir";

    public static final int NN_BAGGING_THRESHOLD = 50;

    public static final String NN_POISON_SAMPLER = "nn.poison.sampler.enable";

    public static final double DRY_ERROR = 0.0d;

    public static final int DEFAULT_EPOCHS_PER_ITERATION = 1;

    public static final String DEFAULT_GUAGUA_VERSION = "0.1.0";

    public static final int DEFAULT_JOIN_TIME = 3000;
    
    public static final double DEFAULT_SIGNIFICANCE_VALUE = 1.0;
}
