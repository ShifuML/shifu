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
from tensorflow.python.platform import gfile
import gzip
from StringIO import StringIO
import random
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import numpy as np
import sys
import os
import datetime


def tprint(content, log_level="INFO"):
    systime = datetime.datetime.now()
    print(str(systime) + " " + log_level + " " + " [Shifu.Tensorflow.train] " + str(content))


def load_data(context):

    train_data = []
    train_target = []
    valid_data = []
    valid_target = []

    file_count = 0
    train_pos_cnt = 0
    train_neg_cnt = 0
    valid_pos_cnt = 0
    valid_neg_cnt = 0

    feature_column_nums = context["feature_column_nums"]

    allFileNames = gfile.ListDirectory(root)
    normFileNames = filter(lambda x: not x.startswith(".") and not x.startswith("_"), allFileNames)
    tprint(normFileNames)
    tprint("Total input file count is " + str(len(normFileNames)) + ".")

    line_count = 0
    for normFileName in normFileNames:
        tprint("Now loading " + normFileName + " Progress: " + str(file_count) + "/" + str(len(normFileNames)) + ".")
        file_count += 1
        with gfile.Open(root + '/' + normFileName, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            while True:
                line = gf.readline()
                if len(line) == 0:
                    break
                line_count += 1
                columns = line.split(delimiter)

                if feature_column_nums == None:
                    feature_column_nums = range(0, len(columns))
                    feature_column_nums.remove(target_index)

                if random.random() >= valid_data_percentage:
                    # Append training data
                    train_target.append([float(columns[target_index])])
                    if(columns[target_index] == "1"):
                        train_pos_cnt += 1
                    else :
                        train_neg_cnt += 1
                    single_train_data = []
                    for feature_column_num in feature_column_nums:
                        single_train_data.append(float(columns[feature_column_num].strip('\n')))
                    train_data.append(single_train_data)
                else:
                    # Append validation data
                    valid_target.append([float(columns[target_index])])
                    if(columns[target_index] == "1"):
                        valid_pos_cnt += 1
                    else:
                        valid_neg_cnt += 1
                    single_valid_data = []
                    for feature_column_num in feature_column_nums:
                        single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                    valid_data.append(single_valid_data)
    tprint("Total date count: " + str(line_count) + ".")
    tprint("Train pos count: " + str(train_pos_cnt) + ".")
    tprint("Train neg count: " + str(train_neg_cnt) + ".")
    tprint("Valid pos count: " + str(valid_pos_cnt) + ".")
    tprint("Valid neg count: " + str(valid_neg_cnt) + ".")

    tprint("Data load finished.")

    context['feature_count'] = len(feature_column_nums)

    return train_data, train_target, valid_data, valid_target


def build_graph(shifu_context):
    graph = tf.get_default_graph
    in_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, context["feature_count"]), name="shifu_input_0")
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    layers = shifu_context["layers"]
    current_nodes = shifu_context["feature_count"]
    current_layer = in_placeholder
    dnn_layer = []
    weights = []

    for i in range(len(layers)):
        node_num = layers[i]
        weight = tf.Variable(tf.random_uniform([current_nodes, node_num]))
        bias = tf.Variable(tf.random_normal(shape=(1, 1)))
        current_layer = tf.matmul(current_layer, weight)
        current_layer = tf.add(current_layer, bias)
        current_layer = tf.nn.relu(current_layer)
        weights.append(weight)
        current_nodes = node_num
        dnn_layer.append(current_layer)

    weight = tf.Variable(tf.random_uniform([current_nodes, 1]))
    output_layer = tf.nn.sigmoid(tf.matmul(current_layer, weight), name="shifu_output_0")
    cost_func = tf.nn.l2_loss(label_placeholder - output_layer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost_func)
    validate_error = label_placeholder - output_layer
    validate_error = tf.pow(validate_error, 2)
    tprint("Graph build finished.")
    return output_layer, cost_func, optimizer, in_placeholder, label_placeholder, validate_error, graph


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


def train(input_placeholder, target_placeholder, output_layer, cost_func, optimizer, train_or_validate_error, input_features, targets, validate_input, validate_target, session, context):

    session.run(tf.global_variables_initializer())
    epoch = context["epoch"]
    batch_size = context["batch_size"]
    export_dir = context["export_dir"] + "/" + context["model_name"]
    checkpoint_interval = context["checkpoint_interval"]
    total_batch = int(len(input_features) / batch_size)
    input_batch = np.array_split(input_features, total_batch)
    target_batch = np.array_split(targets, total_batch)
    validate_input = np.array_split(validate_input, 1)
    validate_target = np.array_split(validate_target, 1)
    for i in range(1, epoch + 1):
        tprint("Start epoch " + str(i))
        sum_train_error = 0.0
        for j in range(total_batch):
            o, l, c, e = session.run([optimizer, output_layer, cost_func, train_or_validate_error],
                                  feed_dict={
                                      input_placeholder: input_batch[j],
                                      target_placeholder: target_batch[j],
                                  })
            sum_train_error = reduce(lambda x, y: x + y, reduce(lambda x1, y1: np.append(x1, y1),  e))
        tprint("Epoch " + str(i) + " avg training error is " + str(sum_train_error / len(input_features)) + ".")

        sum_validate_error = 0.0
        for j in range(len(validate_input)):
            v = session.run([train_or_validate_error],
                            feed_dict={
                                input_placeholder: validate_input[j],
                                target_placeholder: validate_target[j],
                            })
            sum_validate_error = reduce(lambda x, y: x + y, reduce(lambda x1, y1: x1.append(y1), v))[0]
        tprint("Epoch " + str(i) + " avg validation error is " + str(sum_validate_error / len(validate_input)) + ".")
        if checkpoint_interval > 0 and i % checkpoint_interval == 0:
            simple_save(session=session, export_dir=export_dir + "-checkpoint-" + str(i),
                        inputs={
                            "shifu_input_0": input_placeholder
                        },
                        outputs ={
                            "shifu_output_0": output_layer
                        })
            tprint("Save checkpoint model at epoch " + str(i))

    simple_save(session=session, export_dir=export_dir,
                               inputs={
                                   "shifu_input_0": input_placeholder
                                },
                               outputs ={
                                   "shifu_output_0": output_layer
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

    args, unknown = parser.parse_known_args()

    root = args.inputdaatapath
    target_index = args.target
    hidden_layers = args.hiddenlayernodes
    feature_column_nums = args.selectedcolumns
    valid_data_percentage = args.validationrate
    model_name = args.modelname
    delimiter = args.delimiter.replace('\\', "")
    context = {"feature_column_nums": feature_column_nums ,"layers": hidden_layers, "batch_size": 10,
               "export_dir": "./models", "epoch": args.epochnums, "model_name": model_name, "checkpoint_interval": args.checkpointinterval}
    if not os.path.exists("./models"):
        os.makedirs("./models", 0777)
    input_features, targets, validate_feature, validate_target = load_data(context)

    output_layer, cost_func, optimizer, input_placeholder, target_placeholder, \
        validate_error, graph = build_graph(shifu_context=context)
    session = tf.Session()
    train(input_placeholder=input_placeholder, target_placeholder=target_placeholder, output_layer=output_layer,
          cost_func=cost_func, optimizer=optimizer, train_or_validate_error=validate_error, input_features=input_features,
          targets=targets, validate_input=validate_feature, validate_target=validate_target, session=session, context=context)
