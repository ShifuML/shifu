"""Synchronous SGD
"""

# from __future__ import print_function
import os
import tensorflow as tf
import argparse
import time
import sys
import logging
import gzip
from StringIO import StringIO
import random
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.estimator import model_fn as model_fn_lib
import json
import socket
import shutil
from tensorflow.python.client import timeline
import datetime
import math
#from threading import Thread
#import tensorboard.main as tb_main

TRAINING_MODE = "Training"
EVAL_MODE = "Validation"
HIDDEN_NODES_COUNT = 20
VALID_TRAINING_DATA_RATIO = 0.1

BUILD_MODEL_BY_CONF_ENABLE = True
REPLICAS_TO_AGGREGATE_RATIO = 1

DELIMITER = '|'
BATCH_SIZE = 128

# read from env
cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
n_pss = len(cluster_spec['ps'])  # the number of parameter servers
n_workers = int(os.environ["WORKER_CNT"])  # the number of worker nodes
job_name = os.environ["JOB_NAME"]
task_index = int(os.environ["TASK_ID"])
socket_server_port = int(os.environ["SOCKET_SERVER_PORT"])  # The port of local java socket server listening, to sync worker training intermediate information with master
total_training_data_number = 40578 #int(os.environ["TOTAL_TRAINING_DATA_NUMBER"]) # total data
feature_column_nums = [int(s) for s in str(os.environ["SELECTED_COLUMN_NUMS"]).split(' ')]  # selected column numbers
FEATURE_COUNT = len(feature_column_nums)

sample_weight_column_num = int(os.environ["WEIGHT_COLUMN_NUM"])  # weight column number, default is -1
target_column_num = int(os.environ["TARGET_COLUMN_NUM"])  # target column number, default is -1

tmp_model_path = os.environ["TMP_MODEL_PATH"]
final_model_path = os.environ["FINAL_MODEL_PATH"]

# This client is used for sync worker training intermediate information with master
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_client.connect(("127.0.0.1", socket_server_port))

#######################################################################################################################
#### Start of Define TF Graph: User can change below graph but make sure tf.train.SyncReplicasOptimizer not changed
#######################################################################################################################
def model(x, y_, sample_weight, model_conf):
    logging.info("worker_num:%d" % n_workers)
    logging.info("total_training_data_number:%d" % total_training_data_number)

    if BUILD_MODEL_BY_CONF_ENABLE and model_conf is not None:
        output_digits, output_nodes = generate_from_modelconf(x, model_conf)
    else:
        output_digits = nn_layer(x, FEATURE_COUNT, HIDDEN_NODES_COUNT, act_op_name="hidden_layer_1")
        output_nodes = HIDDEN_NODES_COUNT

    logging.info("output_nodes : " + str(output_nodes))
    y = nn_layer(output_digits, output_nodes, 1, act=tf.nn.sigmoid, act_op_name="shifu_output_0")

    # count the number of updates
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False,
                                  dtype=tf.int32)

    loss = tf.losses.mean_squared_error(predictions=y, labels=y_, weights=sample_weight)

    # we suppose every worker has same batch_size
    if model_conf is not None:
        learning_rate = model_conf['train']['params']['LearningRate']
    else:
        learning_rate = 0.003
    opt = tf.train.SyncReplicasOptimizer(
        #tf.train.GradientDescentOptimizer(learning_rate),
        #tf.train.AdamOptimizer(learning_rate=learning_rate),
        get_optimizer(model_conf['train']['params']['Propagation'])(learning_rate=learning_rate),
        replicas_to_aggregate=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE * REPLICAS_TO_AGGREGATE_RATIO),
        total_num_replicas=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE),
        name="shifu_sync_replicas")
    train_step = opt.minimize(loss, global_step=global_step)

    return opt, train_step, loss, global_step, y
#######################################################################################################################
#### END of Define TF Graph
#######################################################################################################################

def nn_layer(input_tensor, input_dim, output_dim, l2_scale=0.01, act=tf.nn.tanh, act_op_name=None):
    l2_reg = tf.contrib.layers.l2_regularizer(scale=l2_scale)
    weights = tf.get_variable(name="weight_"+str(act_op_name),
                              shape=[input_dim, output_dim],
                              regularizer=l2_reg,
                              #initializer=tf.glorot_uniform_initializer())
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases_"+str(act_op_name),
                             shape=[output_dim],
                             regularizer=l2_reg,
                             #initializer=tf.glorot_uniform_initializer())
                             initializer=tf.contrib.layers.xavier_initializer())

    activations = act(tf.matmul(input_tensor, weights) + biases, name=act_op_name)
    return activations


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

def get_optimizer(name):
    if 'Adam' == name:
        return tf.train.AdamOptimizer
    elif 'B' == name:
        return tf.train.GradientDescentOptimizer
    elif 'AdaGrad' == name:
        return tf.train.AdagradOptimizer
    else:
        return tf.train.AdamOptimizer

def generate_from_modelconf(x, model_conf):
    train_params = model_conf['train']['params']
    num_hidden_layer = int(train_params['NumHiddenLayers'])
    num_hidden_nodes = [int(s) for s in train_params['NumHiddenNodes']]
    activation_func = [get_activation_fun(s) for s in train_params['ActivationFunc']]
    if "RegularizedConstant" in train_params:
        l2_scale = train_params["RegularizedConstant"]
    else:
        l2_scale = 0.01

    global FEATURE_COUNT
    logging.info("NN information: feature count: %s, hiddern layer: %s, hidden nodes: %s" % (FEATURE_COUNT, num_hidden_layer, str(num_hidden_nodes)))
    
    # first layer
    previous_layer = nn_layer(x, FEATURE_COUNT, num_hidden_nodes[0], l2_scale=l2_scale,
                     act=activation_func[0], act_op_name="hidden_layer_" + str(0))

    for i in range(1, num_hidden_layer):
        layer = nn_layer(previous_layer, num_hidden_nodes[i-1], num_hidden_nodes[i], l2_scale=l2_scale,
                     act=activation_func[i], act_op_name="hidden_layer_" + str(i))
        previous_layer = layer

    return previous_layer, num_hidden_nodes[num_hidden_layer-1]


def get_initalizer(name):
    if 'gaussian' == name:
        return tf.initializers.random_normal()
    elif 'xavier' == name:
        return tf.contrib.layers.xavier_initializer()
    else:
        return tf.contrib.layers.xavier_initializer()

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

def dnn_model_fn(features, labels, mode, params):
    logging.error("features:" + str(features))
    shifu_context = params['shifu_context']
    layers = shifu_context["layers"]
    global FEATURE_COUNT
    FEATURE_COUNT = shifu_context["feature_count"]
    learning_rate = shifu_context["learning_rate"]

    loss_func = shifu_context["loss_func"]
    optimizer_name = shifu_context["optimizer"]
    weight_initalizer = shifu_context["weight_initalizer"]
    act_funcs = shifu_context["act_funcs"]

    input_layer = tf.convert_to_tensor(features['input_feature'], dtype=tf.float32)
    #input_layer = features['input_feature']
    sample_weight = tf.convert_to_tensor(features['sample_weight'], dtype=tf.float32)

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
        opt = tf.train.SyncReplicasOptimizer(
            #tf.train.GradientDescentOptimizer(learning_rate),
            #tf.train.AdamOptimizer(learning_rate=learning_rate),
            get_optimizer(optimizer_name)(learning_rate=learning_rate),
            replicas_to_aggregate=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE * REPLICAS_TO_AGGREGATE_RATIO),
            total_num_replicas=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE),
            name="shifu_sync_replicas")
        train_op = opt.minimize(average_loss, global_step=tf.train.get_global_step())
                    # init ops
        init_tokens_op = opt.get_init_tokens_op()
        # initialize local step
        local_init = opt.local_step_init_op
        sync_replicas_hook = opt.make_session_run_hook(shifu_context["is_chief"])
        if shifu_context["is_chief"]:
            # initializes token queue
            local_init = opt.chief_init_op

        # checks if global vars are init
        ready_for_local_init = opt.ready_for_local_init_op

        # Initializing the variables
        init_op = tf.initialize_all_variables()
        logging.info("---Variables initialized---")
        stop_hook = tf.train.StopAtStepHook(num_steps=shifu_context['epoch'])
        chief_hooks = [sync_replicas_hook, stop_hook]

        return tf.estimator.EstimatorSpec(mode=mode, loss=average_loss, train_op=train_op, training_chief_hooks=chief_hooks, training_hooks=[sync_replicas_hook])

    eval_metrics = {"a-loss": tf.metrics.mean_squared_error(predictions=prediction, labels=labels,
                                                            weights=features['sample_weight'])}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            # Report sum of error for compatibility with pre-made estimators
            loss=average_loss,
            eval_metric_ops=eval_metrics)

class TrainAndEvalErrorHook(tf.train.SessionRunHook):
    _current_epoch = 1

    def __init__(self, mode_name=None, data_cnt=0, batch_size=1):
        self._mode_name = mode_name
        self._data_cnt = float(data_cnt)
        # TODO such steps should be recompute
        self.steps_per_epoch = math.ceil(data_cnt / batch_size)
        self.total_loss = 0.0
        self.current_step = 1
        logging.info("")
        logging.info("*** " + self._mode_name + " Hook: - Created")
        logging.info("steps_per_epoch: " + str(self.steps_per_epoch))
        logging.info("")
    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def before_run(self, run_context):
        graph = run_context.session.graph

        # tensor_name = 'loss_tensor_0'
        # loss_tensor = graph.get_tensor_by_name(tensor_name)

        loss_tensor = graph.get_collection(tf.GraphKeys.LOSSES)[0]
        return tf.train.SessionRunArgs(loss_tensor, self._global_step_tensor)

    def after_run(self, run_context, run_values):
        logging.info("Eval: " + str(run_values));
        current_loss = run_values.results[0]
        self.total_loss += current_loss
        
        global_step = run_values.results[1] + 1

        
        if EVAL_MODE == self._mode_name:
            logging.info("Eval: " +self._mode_name + " Epoch " + str(
                global_step - 1) +  ": Loss :" + str(self.total_loss))
        elif TRAINING_MODE == self._mode_name:
            logging.info("Training" + self._mode_name + " Epoch " + str(global_step-1) + ": Loss :" + str(
                self.total_loss))
        else:
            logging.info("Invalid mode name: " + self._mode_name)
        
        # Send intermediate result to master
        message = "worker_index:{},time:{},current_epoch:{},training_loss:{},valid_loss:{},valid_time:{}\n".format(
            str(task_index), "1", str(global_step), str(self.total_loss), "0", "1")
        if sys.version_info < (3, 0):
            socket_client.send(bytes(message))
        else:
            socket_client.send(bytes(message), 'utf8')

        self.total_loss = 0.0
        if "Training" == self._mode_name:
            type(self)._current_epoch += 1
        
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

def main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    logging.info("job_name:%s, task_index:%d" % (job_name, task_index))

    ps_hosts = cluster_spec['ps']
    worker_hosts = cluster_spec['worker']
    
    myaddr = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    logging.info("myaddr = %s" % myaddr)
    
    chief = worker_hosts[0]
    new_workers = worker_hosts[1:len(worker_hosts)]

    cluster = {'chief': [chief],
               'ps': ps_hosts,
               'worker': worker_hosts}
    new_job_name = 'chief' if (task_index == 0 and job_name == 'worker') else job_name
    
    cluster_task_index = task_index
    if job_name == 'worker' and task_index != 0:  # checks if parameter server
        cluster_task_index -= 1;

    os.environ['TF_CONFIG'] = json.dumps(
                                {'cluster': cluster,
                                 'task': {'type': new_job_name, 'index': task_index}})
    logging.info("TF_CONFIG = %s" % os.environ['TF_CONFIG'])
    if job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=task_index)
        server.join()
    else:  # it must be a worker server
        is_chief = (task_index == 0)  # checks if this is the chief node
        logging.info("Loading data from worker index = %d." % task_index)
    
    TIME_INTERVAL_TO_DO_VALIDATION = 3  # seconds

    logging.info("Loading data from worker index = %d." % task_index)

    training_data_path = os.environ["TRAINING_DATA_PATH"]
    if "TRAINING_DATA_PATH" in os.environ:
        logging.info("This is a normal worker..")
    else:
        logging.info("This is a backup worker")
        # watching certain file in hdfs which contains its training data

    # Read model structure info from ModelConfig
    with open('./ModelConfig.json') as f:
        model_conf = json.load(f)
        logging.info("model" + str(model_conf))
        EPOCH = int(model_conf['train']['numTrainEpochs'])
        global VALID_TRAINING_DATA_RATIO
        VALID_TRAINING_DATA_RATIO = model_conf['train']['validSetRate']
        is_continue_train = model_conf['train']['isContinuous']
        global BATCH_SIZE
        if "MiniBatchs" in model_conf['train']['params']:
            BATCH_SIZE = model_conf['train']['params']['MiniBatchs']

        logging.info("Batch size: %s, VALID_TRAINING_DATA_RATIO: %s." % (str(BATCH_SIZE), str(VALID_TRAINING_DATA_RATIO)))

    # import data
    context = load_data(training_data_path)

    if model_conf is not None:
        learning_rate = model_conf['train']['params']['LearningRate']
    else:
        learning_rate = 0.003

    shifu_context = {
            "feature_column_nums": feature_column_nums, "layers": model_conf['train']['params']['NumHiddenNodes'], "batch_size": BATCH_SIZE, "feature_count": FEATURE_COUNT, "model_name": model_conf['basic']['name'], "is_chief": is_chief, 
            "export_dir": final_model_path, "epoch": EPOCH, "sample_weight_column_num": sample_weight_column_num,
            "learning_rate": learning_rate, "loss_func": model_conf['train']['params']['Loss'], "optimizer": "adam",
            "weight_initalizer": "xavier", "act_funcs": model_conf['train']['params']['ActivationFunc']}

    # Train the model TODO epcoch and step in below 
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input_feature': np.asarray(context['train_data'], dtype=np.float32),
           'sample_weight': np.asarray(context["train_data_sample_weight"], dtype=np.float32)},
        y=np.asarray(context["train_target"], dtype=np.float32),
        batch_size=shifu_context["batch_size"],
        num_epochs=shifu_context['epoch'],
        shuffle=False)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=EPOCH,
                                        hooks=[])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input_feature': np.asarray(context['valid_data'], dtype=np.float32),
           'sample_weight': np.asarray(context["valid_data_sample_weight"], dtype=np.float32)},
        y=np.asarray(context["valid_target"], dtype=np.float32),
        batch_size=len(context["valid_target"]),
        num_epochs=shifu_context['epoch'],
        shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      throttle_secs=TIME_INTERVAL_TO_DO_VALIDATION,
                                      hooks=[TrainAndEvalErrorHook(EVAL_MODE, len(context["valid_target"]), len(context["valid_target"]))])

    run_config = tf.estimator.RunConfig(tf_random_seed=19830610,
                                        model_dir=tmp_model_path,
                                        save_checkpoints_steps=10,
                                        log_step_count_steps=1)
    #train_distribute=tf.contrib.distribute.ParameterServerStrategy)
    dnn = tf.estimator.Estimator(model_fn=dnn_model_fn, params={'shifu_context': shifu_context}, config=run_config)
    tf.estimator.train_and_evaluate(dnn, train_spec, eval_spec)

    if shifu_context['is_chief']:
        export_dir = shifu_context["export_dir"]
        dnn.export_savedmodel(export_dir, serving_input_receiver_fn)
        export_generic_config(export_dir=export_dir)


def load_data(data_file):
    data_file_list = data_file.split(",")
    global feature_column_nums

    logging.info("input data %s" % data_file_list)
    logging.info("Selected columns: " + str(feature_column_nums))

    train_data = []
    train_target = []
    valid_data = []
    valid_target = []

    training_data_sample_weight = []
    valid_data_sample_weight = []

    train_pos_cnt = 0
    train_neg_cnt = 0
    valid_pos_cnt = 0
    valid_neg_cnt = 0

    file_count = 1
    line_count = 0

    for currentFile in data_file_list:
        logging.info(
            "Now loading " + currentFile + " Progress: " + str(file_count) + "/" + str(len(data_file_list)) + ".")
        file_count += 1

        with gfile.Open(currentFile, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            while True:
                line = gf.readline()
                if len(line) == 0:
                    break

                line_count += 1
                if line_count % 10000 == 0:
                    logging.info("Total loading lines: " + str(line_count))

                columns = line.split(DELIMITER)

                if feature_column_nums is None:
                    feature_column_nums = range(0, len(columns))

                    feature_column_nums.remove(target_column_num)
                    if sample_weight_column_num >= 0:
                        feature_column_nums.remove(sample_weight_column_num)

                if random.random() >= VALID_TRAINING_DATA_RATIO:
                    # Append training data
                    train_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1":
                        train_pos_cnt += 1
                    else:
                        train_neg_cnt += 1
                    single_train_data = []
                    for feature_column_num in feature_column_nums:
                        try:
                            single_train_data.append(float(columns[feature_column_num].strip('\n')))
                        except:
                            logging.info("Could not convert " + str(columns[feature_column_num].strip('\n') + " to float"))
                            logging.info("feature_column_num: " + str(feature_column_num))
                    train_data.append(single_train_data)

                    weight = float(columns[len(columns)-1].strip('\n'))
                    if weight < 0.0:
                        logging.info("Warning: weight is below 0. example:" + line)
                        weight = 1.0
                    training_data_sample_weight.append([weight])
                else:
                    # Append validation data
                    valid_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1":
                        valid_pos_cnt += 1
                    else:
                        valid_neg_cnt += 1
                    single_valid_data = []
                    for feature_column_num in feature_column_nums:
                        try:
                            single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                        except:
                            logging.info("Could not convert " + str(columns[feature_column_num].strip('\n') + " to float"))
                            logging.info("feature_column_num: " + str(feature_column_num))

                    valid_data.append(single_valid_data)

                    weight = float(columns[len(columns)-1].strip('\n'))
                    if weight < 0.0:
                        logging.info("Warning: weight is below 0. example:" + line)
                        weight = 1.0
                    valid_data_sample_weight.append([weight])

    logging.info("Total data count: " + str(line_count) + ".")
    logging.info("Train pos count: " + str(train_pos_cnt) + ", neg count: " + str(train_neg_cnt) + ".")
    logging.info("Valid pos count: " + str(valid_pos_cnt) + ", neg count: " + str(valid_neg_cnt) + ".")

    return {"train_data": train_data, "train_target": train_target,
            "valid_data": valid_data, "valid_target": valid_target,
            "train_data_sample_weight": training_data_sample_weight,
            "valid_data_sample_weight": valid_data_sample_weight,
            "feature_count": len(feature_column_nums)}


def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
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

def serving_input_receiver_fn():
    global FEATURE_COUNT
    inputs = {
        'input_feature': tf.placeholder(tf.float32, [None, FEATURE_COUNT], name='shifu_input_0'),
        'sample_weight': tf.placeholder(tf.float32, [None, 1], name='shifu_input_wgt_0')
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

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
    f = tf.gfile.GFile(export_dir + "/GenericModelConfig.json", mode="w+")
    f.write(config_json_str)


def start_tensorboard(checkpoint_dir):
    tf.flags.FLAGS.logdir = checkpoint_dir
    if TB_PORT_ENV_VAR in os.environ:
        tf.flags.FLAGS.port = os.environ['TB_PORT']

    tb_thread = Thread(target=tb_main.run_main)
    tb_thread.daemon = True

    logging.info("Starting TensorBoard with --logdir=" + checkpoint_dir + " in daemon thread...")
    tb_thread.start()

if __name__ == '__main__':
    tf.app.run()