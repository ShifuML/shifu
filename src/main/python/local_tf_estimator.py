# -*- coding: utf-8 -*-

# Copyright [2012-2018] PayPal Software Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Python Tensorflow NN Training, user can change TF DAG to customize NN used for training. Models will be saved in the
# same folder of regular models in 'models' folder and being evaluated in distributed shifu eval step.
#

import argparse
import datetime
import gzip
import math
import os
import random
import shutil
import sys
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

tf.compat.v1.disable_v2_behavior()

FEATURE_CNT = 0
TRAINING_MODE = "Training"
EVAL_MODE = "Validation"


def tprint(content, log_level="INFO"):
    systime = datetime.datetime.now()
    print((str(systime) + " " + log_level + " " + " [Shifu.Tensorflow.train] " + str(content)))
    sys.stdout.flush()


def get_activation_fun(name):
    if name is None:
        return tf.nn.leaky_relu
    name = name.lower()

    if 'sigmoid' == name:
        return tf.nn.sigmoid
    elif 'tanh' == name:
        return tf.nn.tanh
    elif 'relu' == name:
        return tf.nn.relu
    elif 'leakyrelu' == name:
        return tf.nn.leaky_relu
    else:
        return tf.nn.leaky_relu


def get_loss_func(name):
    if name is None:
        return tf.losses.mean_squared_error
    name = name.lower()

    if 'squared' == name:
        return tf.losses.mean_squared_error
    elif 'absolute' == name:
        return tf.compat.v1.losses.absolute_difference
    elif 'log' == name:
        return tf.compat.v1.losses.log_loss
    else:
        return tf.losses.mean_squared_error


def get_optimizer(name):
    if name is None:
        return tf.keras.optimizers.Adam
    name = name.lower()

    if 'adam' == name:
        return tf.keras.optimizers.Adam
    elif 'gradientdescent' == name:
        return tf.keras.optimizers.SGD
    elif 'rmsprop' == name:
        return tf.keras.optimizers.RMSprop
    else:
        return tf.keras.optimizers.Adam


def get_initalizer(name):
    if 'gaussian' == name:
        return tf.random_normal_initializer()
    elif 'xavier' == name:
        return tf.initializers.GlorotUniform()
    else:
        return tf.initializers.GlorotUniform()


def export_generic_config(export_dir):
    config_json_str = ""
    config_json_str += "{\n"
    config_json_str += "    \"inputnames\": [\n"
    config_json_str += "        \"shifu_input_0\"\n"
    config_json_str += "      ],\n"
    config_json_str += "    \"properties\": {\n"
    config_json_str += "         \"algorithm\": \"tensorflow\",\n"
    config_json_str += "         \"tags\": [\"serve\"],\n"
    config_json_str += "         \"outputnames\": \"shifu_output_0\",\n"
    config_json_str += "         \"normtype\": \"ZSCALE\"\n"
    config_json_str += "      }\n"
    config_json_str += "}"
    f = open(export_dir + "/" + "GenericModelConfig.json", mode="w+")
    f.write(config_json_str)


def remove_path(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) and os.path.exists(path):
        os.remove(path)
        return
    files = os.listdir(path)
    for f in files:
        remove_path(path + "/" + f)
    os.removedirs(path)


def load_data(context):
    train_data = []
    train_target = []
    valid_data = []
    valid_target = []

    training_data_sample_weight = []
    valid_data_sample_weight = []

    count = 0
    train_pos_cnt = 0
    train_neg_cnt = 0
    valid_pos_cnt = 0
    valid_neg_cnt = 0

    feature_column_nums = context["feature_column_nums"]
    sample_weight_column_num = context["sample_weight_column_num"]
    allFileNames = gfile.ListDirectory(root)
    normFileNames = [x for x in allFileNames if not x.startswith(".") and not x.startswith("_")]
    print(normFileNames)
    print(("Total input file count is " + str(len(normFileNames)) + "."))
    sys.stdout.flush()

    file_count = 1
    line_count = 0

    for normFileName in normFileNames:
        print(("Now loading " + normFileName + " Progress: " + str(file_count) + "/" + str(len(normFileNames)) + "."))
        sys.stdout.flush()
        file_count += 1

        with gfile.Open(root + '/' + normFileName, 'rb') as f:
            gf = gzip.GzipFile(fileobj=BytesIO(f.read()))
            while True:
                line = gf.readline().decode()
                if len(line) == 0:
                    break

                line_count += 1
                if line_count % 10000 == 0:
                    print(("Total loading lines cnt: " + str(line_count)))
                    sys.stdout.flush()

                columns = line.split(delimiter)

                if feature_column_nums == None:
                    feature_column_nums = list(range(0, len(columns)))
                    feature_column_nums.remove(target_index)

                if random.random() >= valid_data_percentage:
                    # Append training data
                    train_target.append([float(columns[target_index])])
                    if (columns[target_index] == "1"):
                        train_pos_cnt += 1
                    else:
                        train_neg_cnt += 1
                    single_train_data = []
                    for feature_column_num in feature_column_nums:
                        single_train_data.append(float(columns[feature_column_num].strip('\n')))
                    train_data.append(single_train_data)

                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            print(("Warning: weight is below 0. example:" + line))
                            weight = 1.0
                        training_data_sample_weight.append([weight])
                    else:
                        training_data_sample_weight.append([1.0])
                else:
                    # Append validation data
                    valid_target.append([float(columns[target_index])])
                    if (columns[target_index] == "1"):
                        valid_pos_cnt += 1
                    else:
                        valid_neg_cnt += 1
                    single_valid_data = []
                    for feature_column_num in feature_column_nums:
                        single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                    valid_data.append(single_valid_data)

                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            print(("Warning: weight is below 0. example:" + line))
                            weight = 1.0
                        valid_data_sample_weight.append([weight])
                    else:
                        valid_data_sample_weight.append([1.0])

    print(("Total data count: " + str(line_count) + "."))
    print(("Train pos count: " + str(train_pos_cnt) + "."))
    print(("Train neg count: " + str(train_neg_cnt) + "."))
    print(("Valid pos count: " + str(valid_pos_cnt) + "."))
    print(("Valid neg count: " + str(valid_neg_cnt) + "."))
    sys.stdout.flush()

    context['feature_count'] = len(feature_column_nums)

    return train_data, train_target, valid_data, valid_target, training_data_sample_weight, valid_data_sample_weight


def serving_input_receiver_fn():
    global FEATURE_CNT
    inputs = {
        'dense_input': tf.compat.v1.placeholder(tf.float32, [None, FEATURE_CNT], name='shifu_input_0')
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


class TrainAndEvalErrorHook(tf.estimator.SessionRunHook):
    _current_epoch = 1

    def __init__(self, mode_name=None, data_cnt=0, batch_size=1):
        self._mode_name = mode_name
        self._data_cnt = float(data_cnt)
        self.steps_per_epoch = math.ceil(data_cnt / batch_size)
        self.total_loss = 0.0
        self.current_step = 1
        print("")
        print(("*** " + self._mode_name + " Hook: - Created"))
        print(("steps_per_epoch: " + str(self.steps_per_epoch)))
        print("")

    def before_run(self, run_context):

        graph = run_context.session.graph

        # tensor_name = 'loss_tensor_0'
        # loss_tensor = graph.get_tensor_by_name(tensor_name)
        loss_info = graph.get_collection(tf.compat.v1.GraphKeys.LOSSES)
        if len(loss_info) >= 1:
            loss_tensor = loss_info[0]
        else:
            print("error happen")
            loss_tensor = None
        return tf.estimator.SessionRunArgs(loss_tensor)

    def after_run(self, run_context, run_values):
        current_loss = run_values.results
        self.total_loss += current_loss
        if self.current_step >= self.steps_per_epoch:
            if EVAL_MODE == self._mode_name:
                print(("                               " + self._mode_name + " Epoch " + str(
                    type(self)._current_epoch - 1) + ": Loss :" + str(self.total_loss / self._data_cnt)))
            elif TRAINING_MODE == self._mode_name:
                print((self._mode_name + " Epoch " + str(type(self)._current_epoch) + ": Loss :" + str(
                    self.total_loss / self._data_cnt)))
            else:
                print(("invalid mode name: " + self._mode_name))
            sys.stdout.flush()

            self.current_step = 1
            self.total_loss = 0.0
            if "Training" == self._mode_name:
                type(self)._current_epoch += 1
        else:
            self.current_step += 1


def move_model(export_dir):
    if os.path.isfile(export_dir + '/saved_model.pb'):
        os.remove(export_dir + '/saved_model.pb')
    shutil.rmtree(export_dir + "/variables/", ignore_errors=True)

    dirs = [export_dir + "/" + d for d in os.listdir(export_dir) if os.path.isdir(export_dir + "/" + d)]
    latest = sorted(dirs, key=lambda x: os.path.getctime(x), reverse=True)[0]

    for f in os.listdir(latest):
        cur = latest + '/' + f
        if os.path.isdir(cur):
            shutil.copytree(cur, export_dir + '/' + f)
        else:
            shutil.copy(cur, export_dir + '/' + f)


def dnn_model_fn(features, labels, mode, params):
    shifu_context = params['shifu_context']
    layers = shifu_context["layers"]
    global FEATURE_CNT
    FEATURE_CNT = shifu_context["feature_count"]
    learning_rate = shifu_context["learning_rate"]

    loss_func = shifu_context["loss_func"]
    optimizer_name = shifu_context["optimizer"]
    weight_initalizer = shifu_context["weight_initalizer"]
    act_funcs = shifu_context["act_funcs"]

    # print(labels)
    # sys.stdout.flush()

    input_layer = tf.convert_to_tensor(features['dense_input'], dtype=tf.float32)
    # sample_weight = tf.convert_to_tensor(features['sample_weight'], dtype=tf.float32)

    # Start define model structure
    model = [input_layer]
    current_layer = input_layer

    for i in range(len(layers)):
        node_num = layers[i]
        current_layer = tf.layers.dense(inputs=current_layer, units=node_num,
                                        activation=get_activation_fun(act_funcs[i]),
                                        kernel_initializer=get_initalizer(weight_initalizer),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
                                        )
        model.append(current_layer)

    logits = tf.layers.dense(inputs=current_layer, units=1,
                             kernel_initializer=get_initalizer(weight_initalizer),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
                             )

    prediction = tf.nn.sigmoid(logits, name="shifu_output_0")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'scores': prediction
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    average_loss = get_loss_func(loss_func)(predictions=prediction, labels=labels, weights=features['sample_weight'])
    # Pre-made estimators use the total_loss instead of the average,
    # so report total_loss for compatibility.
    # batch_size = tf.shape(labels)[0]
    # total_loss = tf.to_float(batch_size) * average_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(optimizer_name)(learning_rate=learning_rate)
        train_op = optimizer.minimize(average_loss, [])
        return tf.estimator.EstimatorSpec(mode=mode, loss=average_loss, train_op=train_op)

    eval_metrics = {"a-loss": tf.metrics.mean_squared_error(predictions=prediction, labels=labels,
                                                            weights=features['sample_weight'])}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            # Report sum of error for compatibility with pre-made estimators
            loss=average_loss,
            eval_metric_ops=eval_metrics)


def get_model(optimizer_name, _learning_rate, feature_count):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=40, activation='relu', input_shape=(feature_count,)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=get_optimizer(optimizer_name)(learning_rate=_learning_rate),
                  metrics=['mse'])
    return model


if __name__ == "__main__":
    print(("Training input arguments: " + str(sys.argv)))
    sys.stdout.flush()
    # Use for parse Arguments
    parser = argparse.ArgumentParser("Shifu_tensorflow_training")
    parser.add_argument("-inputdaatapath", action='store', dest='inputdaatapath', help="data path used for training",
                        type=str, default='/Users/haifwu/Work/shifu/src/main/python/NormalizedData')
    parser.add_argument("-delimiter", action='store', dest='delimiter', default='|',
                        help="delimiter of data file to seperate columns", type=str)
    parser.add_argument("-target", action='store', dest='target', help="target index in training data file", type=int,
                        default=0)
    parser.add_argument("-validationrate", action='store', dest='validationrate', default=0.2, help="validation rate",
                        type=float)
    parser.add_argument("-hiddenlayernodes", action='store', dest='hiddenlayernodes', help="NN hidden layer nodes",
                        nargs='+', type=int, default=50)
    parser.add_argument("-epochnums", action='store', dest='epochnums', help="", type=int, default=200)
    parser.add_argument("-checkppointinterval", action='store', dest='checkpointinterval', default=20, help="",
                        type=int)
    parser.add_argument("-modelname", action='store', dest='modelname', default="model0", help="", type=str)
    parser.add_argument("-seletectedcolumnnums", action='store', dest='selectedcolumns', help="selected columns list",
                        nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    parser.add_argument("-weightcolumnnum", action='store', dest='weightcolumnnum', help="Sample Weight column num",
                        type=int, default=-1)
    parser.add_argument("-learningRate", action='store', dest='learningRate', help="Learning rate of NN", type=float,
                        default=0.002)

    parser.add_argument("-lossfunc", action='store', dest='lossfunc', help="Loss functions", type=str)
    parser.add_argument("-optimizer", action='store', dest='optimizer', help="optimizer functions", type=str)
    parser.add_argument("-weightinitalizer", action='store', dest='weightinitalizer', help="weightinitalizer functions",
                        type=str)
    parser.add_argument("-actfuncs", action='store', dest='actfuncs', help="act funcs of each hidden layers",
                        nargs='+', type=str, default='tanh')
    parser.add_argument("-minibatch", action='store', dest='minibatch', help="batch size of each iteration", type=int
                        , default=128)
    parser.add_argument("-iscontinuous", action='store', dest='iscontinuous', help="continuous training or not"
                        , type=str, default='False')

    args, unknown = parser.parse_known_args()

    root = args.inputdaatapath
    target_index = args.target
    hidden_layers = args.hiddenlayernodes
    feature_column_nums = args.selectedcolumns
    valid_data_percentage = args.validationrate
    model_name = args.modelname
    delimiter = args.delimiter.replace('\\', "")
    sample_weight_column_num = args.weightcolumnnum
    learning_rate = args.learningRate

    loss_func = args.lossfunc
    optimizer = args.optimizer
    weight_initalizer = args.weightinitalizer
    act_funcs = args.actfuncs
    batch_size = args.minibatch
    is_continuous = args.iscontinuous.upper()

    TIME_INTERVAL_TO_DO_VALIDATION = 3  # seconds

    context = {"feature_column_nums": feature_column_nums, "layers": hidden_layers, "batch_size": batch_size,
               "export_dir": "./models", "epoch": args.epochnums, "model_name": model_name,
               "checkpoint_interval": args.checkpointinterval, "sample_weight_column_num": sample_weight_column_num,
               "learning_rate": learning_rate, "loss_func": loss_func, "optimizer": optimizer,
               "weight_initalizer": weight_initalizer, "act_funcs": act_funcs}
    if not os.path.exists("./models"):
        os.makedirs("./models", 0o777)

    if is_continuous == "TRUE":
        print("Removing previous artifacts...")
        shutil.rmtree('./models/tmp', ignore_errors=True)
    else:
        print("Resuming training...")

    input_features, targets, validate_feature, validate_target, training_data_sample_weight, valid_data_sample_weight = load_data(
        context)
    context["total_steps"] = math.ceil(len(input_features) / context['batch_size']) * context['epoch']

    # tf.logging.set_verbosity(tf.logging.INFO)

    # Train the model
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'dense_input': np.asarray(input_features, dtype=np.float32),
           'sample_weight': np.asarray(training_data_sample_weight, dtype=np.float32)},
        y=np.asarray(targets, dtype=np.float32),
        batch_size=context["batch_size"],
        num_epochs=context['epoch'],
        shuffle=False)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=context["total_steps"],
                                        hooks=[TrainAndEvalErrorHook(TRAINING_MODE, len(input_features),
                                                                     context["batch_size"])])

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'dense_input': np.asarray(validate_feature, dtype=np.float32),
           'sample_weight': np.asarray(valid_data_sample_weight, dtype=np.float32)},
        y=np.asarray(validate_target, dtype=np.float32),
        batch_size=len(validate_target),
        num_epochs=1,
        shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      throttle_secs=TIME_INTERVAL_TO_DO_VALIDATION)

    run_config = tf.estimator.RunConfig(tf_random_seed=19830610,
                                        model_dir='./models/tmp',
                                        save_checkpoints_secs=TIME_INTERVAL_TO_DO_VALIDATION)
    new_model = get_model("adam", learning_rate, len(feature_column_nums))

    dnn = tf.keras.estimator.model_to_estimator(keras_model=new_model)

    # tprint("DEBUG DEBUG DEBUG START")

    # features, target = train_input_fn();
    # estimator_spec = dnn.model_fn(features, target, model_fn_lib.ModeKeys.TRAIN, dnn.config)
    # tprint("DEBUG: loss: "+str(estimator_spec.loss))
    # tprint("DEBUG: train_op: "+str(estimator_spec.train_op))

    # tprint(str(estimator_spec))

    # tprint("DEBUG DEBUG DEBUG end")

    # dnn.train(input_fn=train_input_fn, steps=context['epoch'])
    tf.estimator.train_and_evaluate(dnn, train_spec, eval_spec)

    export_dir = context["export_dir"] + "/" + context["model_name"]
    dnn.export_saved_model(export_dir, serving_input_receiver_fn)

    move_model(export_dir)

    export_generic_config(export_dir=export_dir)

'''
    prediction, cost_func, train_op, input_placeholder, target_placeholder, graph, sample_weight_placeholder = build_graph(shifu_context=context)
    session = tf.Session()
    train(input_placeholder=input_placeholder, target_placeholder=target_placeholder, sample_weight_placeholder = sample_weight_placeholder, prediction=prediction,
          cost_func=cost_func, train_op=train_op, input_features=input_features,
          targets=targets, validate_input=validate_feature, validate_target=validate_target, session=session, context=context,
          training_data_sample_weight=training_data_sample_weight, valid_data_sample_weight=valid_data_sample_weight)
'''
