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
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
from numpy import array
import numpy as np
import os

from StringIO import StringIO
import gzip
import itertools
import datetime
import sys
import random
import math
from abc import ABCMeta, abstractmethod

MODELS_PATH = "./models"


class InputFormat:
    __metaclass__ = ABCMeta

    @abstractmethod
    def next_batch(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @staticmethod
    def get_file_length(file_path):
        with gfile.Open(file_path, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            return sum(1 for line in gf)

    @staticmethod
    def get_split(file_path, start, end):
        lines = []
        with gfile.Open(file_path, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            for line in itertools.islice(gf, start, end):
                lines.append(line)
        return lines

    @staticmethod
    def get_first_line(file_path):
        with gfile.Open(file_path, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            line = gf.readline()
            return line

    @staticmethod
    def tprint(content, log_level="INFO"):
        sys_time = datetime.datetime.now()
        print(str(sys_time) + " " + log_level + " " + " [Shifu.Tensorflow.train] " + str(content))
        sys.stdout.flush()


class FileInputFormat(InputFormat):
    def __init__(self, job_context):
        self._context = job_context
        self._root_folder = job_context["data_root_folder"]
        self._batch_num = job_context["batch_size"]
        self._delimiter = job_context["delimiter"]
        self._feature_column_nums = job_context["feature_column_nums"]
        self._sample_weight_column_num = job_context["sample_weight_column_num"]
        self._target_index = job_context["target_index"]
        self._valid_data_percentage = job_context["valid_data_percentage"]

        # These fields will be initialized on initialize method
        self._file_dict = {}
        self._train_splits = []
        self._current_batch_index = None
        self._batch_size = None
        self._valid_data = []
        self._valid_target = []
        self._valid_data_sample_weight = []

    def get_total_batch(self):
        return len(self._train_splits)

    def next_batch(self):
        """Return batch if have more splits, else return None"""
        if self._current_batch_index is None:
            raise ValueError("FileInputFormat not initialize yet!")
        elif self._current_batch_index >= len(self._train_splits):
            return None

        train_data = []
        train_target = []
        training_data_sample_weight = []

        line_count = 0
        train_pos_cnt = 0
        train_neg_cnt = 0
        for line in self.get_split(*self._train_splits[self._current_batch_index]):
            if line_count % 10000 == 0:
                self.tprint("Total loading lines: " + str(line_count))
            line_count += 1
            columns = line.split(self._delimiter)
            if self._feature_column_nums is None:
                self._feature_column_nums = range(0, len(columns))
                self._feature_column_nums.remove(self._target_index)

            # Append training data
            train_target.append([float(columns[self._target_index])])
            if columns[self._target_index] == "1":
                train_pos_cnt += 1
            else:
                train_neg_cnt += 1
            single_train_data = []
            for feature_column_num in self._feature_column_nums:
                single_train_data.append(float(columns[feature_column_num].strip('\n')))
            train_data.append(single_train_data)

            if 0 <= self._sample_weight_column_num < len(columns):
                weight = float(columns[self._sample_weight_column_num].strip('\n'))
                if weight < 0.0:
                    self.tprint("Warning: weight is below 0. example:" + line)
                    weight = 1.0
                training_data_sample_weight.append([weight])
            else:
                training_data_sample_weight.append([1.0])
        self.tprint("Total data count: " + str(line_count) + ".")
        self.tprint("Train pos count: " + str(train_pos_cnt) + ", neg count: " + str(train_neg_cnt) + ".")
        self._current_batch_index += 1
        return train_data, train_target, self._valid_data, self._valid_target, training_data_sample_weight, self._valid_data_sample_weight

    def initialize(self):
        all_files = gfile.ListDirectory(self._root_folder)
        norm_files = filter(lambda x: not x.startswith(".") and not x.startswith("_"), all_files)
        self.tprint(norm_files)
        self.tprint("Total input file count is " + str(len(norm_files)) + ".")
        sys.stdout.flush()

        file_count = 0
        line_count = 0
        for normal_file in norm_files:
            self.tprint("Now loading " + normal_file + " Progress: " + str(file_count) + "/" + str(len(norm_files)) +
                        ".")
            file_path = os.path.join(self._root_folder, normal_file)
            file_line_cnt = self.get_file_length(file_path)
            self._file_dict[file_path] = file_line_cnt
            sys.stdout.flush()
            file_count += 1
            line_count += file_line_cnt
        self.tprint("Total data files: " + str(file_count) + ".")
        self.tprint("Total data count: " + str(line_count) + ".")
        sys.stdout.flush()

        # Set batch size value according to total data count and batch number
        # this will make sure batch num > 1 if self._valid_data_percentage > 0
        batch_num = int(math.ceil(self._batch_num * 1.0 / (1 - self._valid_data_percentage)))
        self._batch_size = int(math.ceil(line_count * 1.0 / batch_num))
        self.tprint("Total batch num: " + str(batch_num) + ", batch size" + str(self._batch_size) + ".")
        sys.stdout.flush()
        self._current_batch_index = 0

        # Read data sample to calculate feature_count and set in context
        if len(norm_files) > 0:
            first_file_path = os.path.join(self._root_folder, norm_files[0])
            first_line = self.get_first_line(first_file_path)
            columns = first_line.split(self._delimiter)
            if self._feature_column_nums is None:
                self._feature_column_nums = range(0, len(columns))
                self._feature_column_nums.remove(self._target_index)
        self._context["feature_count"] = len(self._feature_column_nums)

        # Initial file splits
        full_splits = []
        evaluate_splits = []
        for file_path, file_line_cnt in self._file_dict.items():
            full_splits.extend(self.__split_file(file_path, file_line_cnt))
        for file_split in full_splits:
            if random.random() >= self._valid_data_percentage:
                self._train_splits.append(file_split)
            else:
                evaluate_splits.append(file_split)

        if len(evaluate_splits) == 0:
            evaluate_splits = self._train_splits[0]
            del self._train_splits[0]
        self.__load_evaluate_date(evaluate_splits)
        self.tprint("Finished initialize training splits: " + "".join([str(split) for split in self._train_splits]))
        self.tprint("Finished initialize evaluate splits: " + "".join([str(split) for split in evaluate_splits]))

    def __load_evaluate_date(self, evaluate_splits):
        for i in range(len(evaluate_splits)):
            line_count = 0
            valid_pos_cnt = 0
            valid_neg_cnt = 0
            for line in self.get_split(*evaluate_splits[i]):
                line_count += 1
                columns = line.split(self._delimiter)
                self._valid_target.append([float(columns[self._target_index])])
                if columns[self._target_index] == "1":
                    valid_pos_cnt += 1
                else:
                    valid_neg_cnt += 1
                single_valid_data = []
                for feature_column_num in self._feature_column_nums:
                    single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                self._valid_data.append(single_valid_data)

                if 0 <= self._sample_weight_column_num < len(columns):
                    weight = float(columns[self._sample_weight_column_num].strip('\n'))
                    if weight < 0.0:
                        self.tprint("Warning: weight is below 0. example:" + line)
                        weight = 1.0
                    self._valid_data_sample_weight.append([weight])
                else:
                    self._valid_data_sample_weight.append([1.0])
            self.tprint("Total loading lines: " + str(line_count))

    def __split_file(self, file_path, total_count):
        splits = []
        batch_number = int(math.ceil(total_count * 1.0 / self._batch_size))
        for i in range(batch_number):
            start = i * self._batch_size
            stop = total_count if (i + 1) * self._batch_size > total_count else (i + 1) * self._batch_size
            splits.append((file_path, start, stop))
        return splits


def tprint(content, log_level="INFO"):
    systime = datetime.datetime.now()
    print(str(systime) + " " + log_level + " " + " [Shifu.Tensorflow.train] " + str(content))
    sys.stdout.flush()

########################################################################################################################
# Buid your own training graph, user can edit tf graph in below method, this case, configuration in ModelConfig.json
# will not be honoured.
########################################################################################################################
def build_graph(shifu_context):
    is_continuous = shifu_context["is_continuous"]
    if is_continuous == "TRUE":
        tf.saved_model.loader.load(session, [tag_constants.SERVING], export_dir=os.path.join(MODELS_PATH, model_name))
        graph = tf.get_default_graph()
        in_placeholder = graph.get_tensor_by_name('shifu_input_0:0')
    else:
        graph = tf.get_default_graph()
        in_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, context["feature_count"]), name="shifu_input_0")

    label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, 1))
    sample_weight_placeholder = tf.placeholder(dtype=tf.float32, shape=(None))

    layers = shifu_context["layers"]
    current_nodes = shifu_context["feature_count"]
    learning_rate = shifu_context["learning_rate"]

    loss_func = shifu_context["loss_func"]
    optimizer_name = shifu_context["optimizer"]
    weight_initalizer = shifu_context["weight_initalizer"]
    act_funcs = shifu_context["act_funcs"]

    tprint("Configuration info: loss func=" + str(loss_func) + " optimizer_name=" + str(optimizer_name) + " weight_initalizer=" + str(weight_initalizer) + " act_funcs=" + str(act_funcs))

    current_layer = in_placeholder
    dnn_layer = []
    weights = []
    biases = []

    l2_reg = tf.contrib.layers.l2_regularizer(scale=0.01)

    for i in range(len(layers)):
        node_num = layers[i]
        weight = tf.Variable(tf.random_uniform([current_nodes, node_num], -1.0, 1.0))
        #weight = tf.get_variable(name="weight_" + str(i), regularizer = l2_reg, initializer= tf.random_uniform([current_nodes, node_num], -1.0, 1.0))

        bias = tf.Variable(tf.random_uniform(shape=([node_num]), minval=-1.0, maxval=1.0))
        current_layer = tf.matmul(current_layer, weight)
        current_layer = tf.add(current_layer, bias)
        current_layer = get_activation_fun(act_funcs[i])(current_layer)
        weights.append(weight)
        biases.append(bias)
        current_nodes = node_num
        dnn_layer.append(current_layer)

    weight = tf.Variable(tf.random_uniform([current_nodes, 1], -1.0, 1.0))
    #weight = tf.get_variable(name="weight_" + str(len(layers)), regularizer = l2_reg, initializer= tf.random_uniform([current_nodes, 1], -1.0, 1.0))
    bias = tf.Variable(tf.random_uniform(shape=([1]), minval=-1.0, maxval=1.0))
    output_layer = tf.matmul(current_layer, weight)
    output_layer = tf.add(output_layer, bias)
    weights.append(weight)
    biases.append(bias)
    dnn_layer.append(output_layer)

    #prediction = tf.cast(tf.argmax(tf.nn.softmax(output_layer), 1), tf.float32, name="shifu_output_0")
    prediction = tf.nn.sigmoid(output_layer, name="shifu_output_0")

    # Define loss and optimizer
    reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights)

    cost_func = (get_loss_func(loss_func)(predictions=prediction, labels=label_placeholder, weights=sample_weight_placeholder) + reg_term)
    #cost_func = get_loss_func(loss_func)(predictions=prediction, labels=label_placeholder, weights=sample_weight_placeholder)

    optimizer = get_optimizer(optimizer_name)(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost_func)

    return prediction, cost_func, train_op, in_placeholder, label_placeholder, graph, sample_weight_placeholder
########################################################################################################################
# End of build tensorflow training graph
########################################################################################################################

def get_activation_fun(name):
    if name == None:
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
    if name == None:
        return tf.losses.mean_squared_error
    name = name.lower()

    if 'squared' == name:
        return tf.losses.mean_squared_error
    elif 'absolute' == name:
        return tf.losses.absolute_difference
    elif 'log' == name:
        return tf.losses.log_loss
    else:
        return tf.losses.mean_squared_error

def get_optimizer(name):
    if name == None:
        return tf.train.AdamOptimizer
    name = name.lower()

    if 'adam' == name:
        return tf.train.AdamOptimizer
    elif 'gradientdescent' == name:
        return tf.train.GradientDescentOptimizer
    elif 'rmsprop' == name:
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.AdamOptimizer

def get_initalizer(name):
    if 'gaussian' == name:
        return tf.initializers.random_normal()
    elif 'xavier' == name:
        return tf.contrib.layers.xavier_initializer()
    else:
        return tf.contrib.layers.xavier_initializer()

def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
    remove_path(export_dir)
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_def_utils.predict_signature_def(inputs, outputs)
    }
    b = builder.SavedModelBuilder(export_dir)
    b.add_meta_graph_and_variables(
        session,
        tags=[tag_constants.SERVING],
        signature_def_map=signature_def_map,
        assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        legacy_init_op=legacy_init_op,
        clear_devices=True)
    b.save()
    export_generic_config(export_dir=export_dir)

def one_hot(input, num_classes):
    input = np.array(input).reshape(-1)
    one_hot_targets = np.eye(num_classes)[input]
    return one_hot_targets

def train(input_placeholder, target_placeholder, sample_weight_placeholder, prediction, cost_func, train_op, session,
          context, input_format):
    session.run(tf.global_variables_initializer())
    epoch = context["epoch"]
    export_dir = os.path.join(context["export_dir"],context["model_name"])
    checkpoint_interval = context["checkpoint_interval"]

    for i in range(1, epoch + 1):
        tprint("Start epoch " + str(i))
        sum_train_error = 0.0

        mini_batch = input_format.next_batch()
        while mini_batch is not None:
            input_batch = array(mini_batch[0])
            target_batch = array(mini_batch[1])
            validate_input = array(mini_batch[2])
            validate_target = array(mini_batch[3])
            train_sample_weight_batch = array(mini_batch[4])
            valid_data_sample_weight = array(mini_batch[5])

            o, c, p = session.run([train_op, cost_func, prediction],
                                  feed_dict={
                                      input_placeholder: input_batch,
                                      target_placeholder: target_batch,
                                      sample_weight_placeholder: train_sample_weight_batch,
                                  })
            sum_train_error += c

            sum_validate_error = 0.0
            v = session.run([cost_func],
                            feed_dict={
                                input_placeholder: validate_input,
                                target_placeholder: validate_target,
                                sample_weight_placeholder: valid_data_sample_weight,
                            })
            sum_validate_error += v[0]
            mini_batch = input_format.next_batch()
        tprint("Epoch " + str(i) + " avg train error " + str(sum_train_error / len(input_batch)) +
               ", avg validation error is " + str(sum_validate_error / len(validate_input)) + ".")

        if checkpoint_interval > 0 and i % checkpoint_interval == 0:
            simple_save(session=session, export_dir=export_dir + "-checkpoint-" + str(i),
                        inputs={
                            "shifu_input_0": input_placeholder
                        },
                        outputs ={
                            "shifu_output_0": prediction
                        })
            tprint("Save checkpoint model at epoch " + str(i))

    simple_save(session=session, export_dir=export_dir,
                inputs={
                    "shifu_input_0": input_placeholder
                },
                outputs ={
                    "shifu_output_0": prediction
                })
    tprint("Model training finished, model export path: " + export_dir)


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
    f = file(export_dir + "/" + "GenericModelConfig.json", mode="w+")
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

if __name__ == "__main__":
    tprint("Training input arguments: " + str(sys.argv))
    # Use for parse Arguments
    parser = argparse.ArgumentParser("Shifu_tensorflow_training")
    parser.add_argument("-inputdaatapath", action='store', dest='inputdaatapath', help="data path used for training",
                        type=str)
    parser.add_argument("-delimiter", action='store', dest='delimiter',
                        help="delimiter of data file to seperate columns", type=str)
    parser.add_argument("-target", action='store', dest='target', help="target index in training data file", type=int)
    parser.add_argument("-validationrate", action='store', dest='validationrate', default=0.2, help="validation rate",
                        type=float)
    parser.add_argument("-hiddenlayernodes", action='store', dest='hiddenlayernodes', help="NN hidden layer nodes",
                        nargs='+',type=int)
    parser.add_argument("-epochnums", action='store', dest='epochnums', help="", type=int)
    parser.add_argument("-checkppointinterval", action='store', dest='checkpointinterval', default=0, help="", type=int)
    parser.add_argument("-modelname", action='store', dest='modelname', default="model0", help="", type=str)
    parser.add_argument("-seletectedcolumnnums", action='store', dest='selectedcolumns', help="selected columns list",
                        nargs='+', type=int)
    parser.add_argument("-weightcolumnnum", action='store', dest='weightcolumnnum', help="Sample Weight column num", type=int)
    parser.add_argument("-learningRate", action='store', dest='learningRate', help="Learning rate of NN", type=float)

    parser.add_argument("-lossfunc", action='store', dest='lossfunc', help="Loss functions", type=str)
    parser.add_argument("-optimizer", action='store', dest='optimizer', help="optimizer functions", type=str)
    parser.add_argument("-weightinitalizer", action='store', dest='weightinitalizer', help="weightinitalizer functions", type=str)
    parser.add_argument("-actfuncs", action='store', dest='actfuncs', help="act funcs of each hidden layers",
                        nargs='+',type=str)
    parser.add_argument("-minibatch", action='store', dest='minibatch', help="batch size of each iteration", type=int)
    parser.add_argument("-iscontinuous", action='store', dest='iscontinuous', help="continuous training or not", default=False)

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

    context = {"feature_column_nums": feature_column_nums, "layers": hidden_layers, "batch_size": batch_size,
               "export_dir": MODELS_PATH, "epoch": args.epochnums, "model_name": model_name,
               "checkpoint_interval": args.checkpointinterval, "sample_weight_column_num": sample_weight_column_num,
               "learning_rate": learning_rate, "loss_func": loss_func, "optimizer": optimizer,
               "weight_initalizer": weight_initalizer, "act_funcs":act_funcs, "is_continuous": is_continuous,
               "data_root_folder": root, "target_index": target_index, "valid_data_percentage": valid_data_percentage,
               "delimiter": delimiter}
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH, 0777)

    session = tf.Session()

    # Init input format
    input_format = FileInputFormat(context)
    input_format.initialize()

    # Train the model
    prediction, cost_func, train_op, input_placeholder, target_placeholder, graph, sample_weight_placeholder = build_graph(shifu_context=context)

    train(input_placeholder=input_placeholder, target_placeholder=target_placeholder,
          sample_weight_placeholder = sample_weight_placeholder, prediction=prediction,
          cost_func=cost_func, train_op=train_op, session=session, context=context,
          input_format=input_format)
