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
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from file_input_format import FileInputFormat
import tensorflow as tf
import numpy as np
from numpy import array
import sys
import os
import datetime

MODELS_PATH = "./models"


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
    num_classes = 2
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
                                sample_weight_placeholder: [valid_data_sample_weight],
                            })
            sum_validate_error += v[0]
            tprint("Epoch " + str(i) + " avg train error " + str(sum_train_error / input_format.get_total_batch()) +
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
               "data_root_folder": root, "target_index": target_index, "valid_data_percentage": valid_data_percentage}
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

