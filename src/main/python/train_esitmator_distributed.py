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
import shutil
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

SHIFU_CONTEXT = {}

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
        return tf.metrics.mean_squared_error
    name = name.lower()

    if 'squared' == name:
        return tf.metrics.mean_squared_error
    elif 'absolute' == name:
        return tf.metrics.mean_absolute_error
    elif 'log' == name:
        # No log error, we use root error here
        return tf.metrics.root_mean_squared_error
    else:
        return tf.metrics.mean_squared_error
    
def get_optimizer(name):
    if name == None:
        return tf.train.AdamOptimizer
    name = name.lower()
    
    if 'adam' == name:
        return tf.train.AdamOptimizer
    elif 'gradientDescent' == name:
        return tf.train.GradientDescentOptimizer
    elif 'RMSProp' == name:
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
    delimiter = context["delimiter"]
    allFileNames = gfile.ListDirectory(root)
    normFileNames = filter(lambda x: not x.startswith(".") and not x.startswith("_"), allFileNames)
    print(normFileNames)
    print("Total input file count is " + str(len(normFileNames)) + ".")
    sys.stdout.flush()

    file_count = 1
    line_count = 0

    for normFileName in normFileNames:
        print("Now loading " + normFileName + " Progress: " + str(file_count) + "/" + str(len(normFileNames)) + ".")
        sys.stdout.flush()
        file_count += 1

        with gfile.Open(root + '/' + normFileName, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            while True:
                line = gf.readline()
                if len(line) == 0:
                    break
                
                line_count += 1
                if line_count % 10000 == 0: 
                    print("Total loading lines cnt: " + str(line_count))
                    sys.stdout.flush()
                
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
                    
                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            print("Warning: weight is below 0. example:" + line)
                            weight= 1.0
                        training_data_sample_weight.append([weight])
                    else:
                        training_data_sample_weight.append([1.0])
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
                    
                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            print("Warning: weight is below 0. example:" + line)
                            weight= 1.0
                        valid_data_sample_weight.append([weight])
                    else:
                        valid_data_sample_weight.append([1.0])

    print("Total data count: " + str(line_count) + ".")
    print("Train pos count: " + str(train_pos_cnt) + ".")
    print("Train neg count: " + str(train_neg_cnt) + ".")
    print("Valid pos count: " + str(valid_pos_cnt) + ".")
    print("Valid neg count: " + str(valid_neg_cnt) + ".")
    sys.stdout.flush()

    context['feature_count'] = len(feature_column_nums)

    return train_data, train_target, valid_data, valid_target, training_data_sample_weight, valid_data_sample_weight

def serving_input_receiver_fn():
    global SHIFU_CONTEXT
    inputs = {
        'input_feature': tf.placeholder(tf.float32, [None, SHIFU_CONTEXT["feature_count"]], name='shifu_input_0')
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def dnn_model_fn(features, labels, mode, params):
    shifu_context = params.shifu_context
    layers = shifu_context["layers"]
    learning_rate = shifu_context["learning_rate"]
    
    loss_func = shifu_context["loss_func"]
    optimizer_name = shifu_context["optimizer"]
    weight_initalizer = shifu_context["weight_initalizer"]
    act_funcs = shifu_context["act_funcs"]

    #print(labels)
    #sys.stdout.flush()
    
    input_layer = tf.convert_to_tensor(features['input_feature'], dtype=tf.float32)
    #sample_weight = tf.convert_to_tensor(features['sample_weight'], dtype=tf.float32)
    
    # Start define model structure
    model = [input_layer]
    current_layer = input_layer
    
    for i in range(len(layers)):
        node_num = layers[i]
        current_layer = tf.layers.dense(inputs=current_layer, units=node_num, activation=get_activation_fun(act_funcs[i]), kernel_initializer=get_initalizer(weight_initalizer))
        model.append(current_layer)
    
    logits = tf.layers.dense(inputs=current_layer, units=1)
    
    prediction = tf.nn.sigmoid(logits, name="shifu_output_0")
    
    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        optimizer = get_optimizer(optimizer_name)(learning_rate=learning_rate)

        # Create training operation
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return train_op

    head = tf.contrib.estimator.regression_head(
            label_dimension=1,
            name='regression_head',
            loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
            loss_fn=get_loss_func(loss_func),
            weight_column='sample_weight'
        )

    return head.create_estimator_spec(
            features,
            mode,
            prediction,
            labels=labels,
            train_op_fn=_train_op_fn
        )

def metric_fn(labels, predictions, features, config):
    metrics = {}

    pred_values = predictions['predictions']
    
    global SHIFU_CONTEXT
    metrics["average_loss"] = get_loss_func(SHIFU_CONTEXT["loss_func"])(labels, pred_values, weights=features['sample_weight'])

    return metrics

def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(
        model_fn=dnn_model_fn, 
        config=run_config,
        params=hparams
    )
    
    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)
    
    return estimator


if __name__ == "__main__":
    print("Training input arguments: " + str(sys.argv))
    print(tf.__version__)
    sys.stdout.flush()
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
    
    RESUME_TRAINING = False
    
    # Make SHIFU_CONTEXT to be global so that metric_fn can be read
    global SHIFU_CONTEXT
    SHIFU_CONTEXT["feature_column_nums"] = feature_column_nums
    SHIFU_CONTEXT["layers"] = hidden_layers
    SHIFU_CONTEXT["batch_size"] = batch_size
    SHIFU_CONTEXT["export_dir"] = "./models"
    SHIFU_CONTEXT["epoch"] = args.epochnums
    SHIFU_CONTEXT["model_name"] = model_name
    SHIFU_CONTEXT["checkpoint_interval"] = args.checkpointinterval
    SHIFU_CONTEXT["sample_weight_column_num"] = sample_weight_column_num
    SHIFU_CONTEXT["learning_rate"] = learning_rate
    SHIFU_CONTEXT["loss_func"] = loss_func
    SHIFU_CONTEXT["optimizer"] = optimizer
    SHIFU_CONTEXT["weight_initalizer"] = weight_initalizer
    SHIFU_CONTEXT["act_funcs"] = act_funcs
    SHIFU_CONTEXT["delimiter"] = delimiter
    
    if not os.path.exists("./models"):
        os.makedirs("./models", 0777)

    input_features, targets, validate_feature, validate_target, training_data_sample_weight, valid_data_sample_weight = load_data(SHIFU_CONTEXT)

    # Train the model
    SHIFU_CONTEXT["total_steps"] = (len(input_features)/SHIFU_CONTEXT['batch_size'])*SHIFU_CONTEXT['epoch']
    export_dir = SHIFU_CONTEXT["export_dir"] + "/" + SHIFU_CONTEXT["model_name"]
    hparams  = tf.contrib.training.HParams(shifu_context=SHIFU_CONTEXT)
    run_config = tf.estimator.RunConfig(tf_random_seed=19830610, model_dir=export_dir)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input_feature': np.asarray(input_features, dtype=np.float32), 'sample_weight': np.asarray(training_data_sample_weight, dtype=np.float32)},
        y=np.asarray(targets, dtype=np.float32),
        batch_size=SHIFU_CONTEXT["batch_size"],
        num_epochs=SHIFU_CONTEXT['epoch'],
        shuffle=False)
    train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn, max_steps=SHIFU_CONTEXT["total_steps"])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input_feature': np.asarray(validate_feature, dtype=np.float32), 'sample_weight': np.asarray(valid_data_sample_weight, dtype=np.float32)},
        y=np.asarray(validate_target, dtype=np.float32),
        batch_size=SHIFU_CONTEXT["batch_size"],
        num_epochs=1,
        shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn)
    
    if not RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(export_dir, ignore_errors=True)
    else:
        print("Resuming training...") 
    sys.stdout.flush()
        
    tf.logging.set_verbosity(tf.logging.INFO)
    
    estimator = create_estimator(run_config, hparams)
    
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec,  eval_spec=eval_spec)

    estimator.export_savedmodel(export_dir, serving_input_receiver_fn)
    export_generic_config(export_dir=export_dir)
    
'''
    prediction, cost_func, train_op, input_placeholder, target_placeholder, graph, sample_weight_placeholder = build_graph(shifu_context=context)
    session = tf.Session()
    train(input_placeholder=input_placeholder, target_placeholder=target_placeholder, sample_weight_placeholder = sample_weight_placeholder, prediction=prediction,
          cost_func=cost_func, train_op=train_op, input_features=input_features,
          targets=targets, validate_input=validate_feature, validate_target=validate_target, session=session, context=context,
          training_data_sample_weight=training_data_sample_weight, valid_data_sample_weight=valid_data_sample_weight)
'''

