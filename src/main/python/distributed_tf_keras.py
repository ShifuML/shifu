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
# ==============================================================================

'''Distributed TF model training class, model definition is keras model.'''

import os
import tensorflow as tf
import argparse
import time
import sys
import logging
import gzip
import math
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
from tensorflow import keras
import json
import socket
from tensorflow.python.client import timeline

#######################################################################################################################
#### Start: Define TF Keras Model: customize below keras model, make sure Inputs and predictions are not changed.
#######################################################################################################################
def model(shifu_context):
    inputs = keras.Input(shape=(shifu_context['feature_count'],), name='shifu_input_0')  # Returns a placeholder tensor

    # such model arch in below can be defined manually rather than use configurations from ModelConfig.json
    train_params = shifu_context['model_conf']['train']['params']
    num_hidden_layer = int(train_params['NumHiddenLayers'])
    num_hidden_nodes = [int(s) for s in train_params['NumHiddenNodes']]
    activation_func = [get_activation_fun(s) for s in train_params['ActivationFunc']]

    previous_layer = inputs
    for i in range(0, num_hidden_layer):
        acti = train_params['ActivationFunc'][i]
        kernel_regularizer = None
        if "RegularizedConstant" in train_params:
            kernel_regularizer = keras.regularizers.l2(l=train_params["RegularizedConstant"])
        kernel_initializer = "glorot_uniform"
        if "WeightInitializer" in train_params:
            kernel_initializer = train_params["WeightInitializer"]

        layer = keras.layers.Dense(num_hidden_nodes[i], activation=acti, name='hidden_layer_'+str(i+1), kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)(previous_layer)
        previous_layer = layer
    predictions = keras.layers.Dense(1, activation='sigmoid', name='shifu_output_0')(layer)

    model = tf.keras.models.Model(inputs, predictions)

    # loss now only string loss supported, loss function not supported now TODO support loss function
    opti = shifu_context['model_conf']['train']['params']['Propagation']; # 'adam', 'sgd' and 'adagrad' are supported
    model.compile(loss='binary_crossentropy', optimizer=get_optimizer(opti)(learning_rate=shifu_context['learning_rate']), metrics=['mse'])
    return model
#######################################################################################################################
#### END: Define TF Keras Model
#######################################################################################################################

def get_optimizer(name):
    name = name.lower()
    if 'adam' == name:
        return tf.train.AdamOptimizer
    elif 'b' == name or 'sgd' == name or 'gd' == name or 'gradientdescent' == name:
        return tf.train.GradientDescentOptimizer
    elif 'adagrad' == name:
        return tf.train.AdagradOptimizer
    else:
        return tf.train.AdamOptimizer

def get_loss_func(name):
    if name == None:
        logging.warn("Loss 'name' is not specidied, set to mean_squared_error.")
        return tf.losses.mean_squared_error
    name = name.lower()

    if 'squared' == name or 'mse' == name or 'mean_squared_error' == name:
        return tf.losses.mean_squared_error
    elif 'absolute' == name:
        return tf.losses.absolute_difference
    elif 'log' == name:
        return tf.losses.log_loss
    elif 'binary_crossentropy' == name:
        return tf.losses.log_loss
    else:
        return tf.losses.mean_squared_error

def get_activation_fun(name):
    if name is None:
        return tf.nn.relu
    name = name.lower()

    if 'sigmoid' == name:
        return tf.nn.sigmoid
    elif 'tanh' == name:
        return tf.nn.tanh
    elif 'relu' == name:
        return tf.nn.relu
    elif 'leakyrelu' == name:
        return tf.nn.leaky_relu
    elif 'leaky_relu' == name:
        return tf.nn.leaky_relu
    else:
        return tf.nn.relu

def read_context_from_env_and_modelconf():
    replicas_to_aggregate_ratio = 1 # Aggregation replica reatio, default is 1, setting to < 1 can accerlerate traning but accuracy may be dropped.
    
    delimiter = '|'
    if "DELIMITER" in os.environ:
        delimiter = os.environ['DELIMITER']

    # Read properties from ENV
    cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
    n_pss = len(cluster_spec['ps'])  # the number of parameter servers
    n_workers = int(os.environ["WORKER_CNT"])  # the number of worker nodes
    job_name = os.environ["JOB_NAME"]
    task_index = int(os.environ["TASK_ID"])
    is_chief = (job_name == "worker" and task_index == 0)
    socket_server_port = int(os.environ["SOCKET_SERVER_PORT"])  # The port of local java socket server listening, to sync worker training intermediate information with master
    total_training_data_number = int(os.environ["TOTAL_TRAINING_DATA_NUMBER"]) # total data 200468
    feature_column_nums = [int(s) for s in str(os.environ["SELECTED_COLUMN_NUMS"]).split(' ')]  # selected column numbers
    feature_count = len(feature_column_nums) # number of input columns

    sample_weight_column_num = int(os.environ["WEIGHT_COLUMN_NUM"])  # weight column number, default is -1
    target_column_num = int(os.environ["TARGET_COLUMN_NUM"])  # target column number, default is -1

    tmp_model_path = os.environ["TMP_MODEL_PATH"]
    final_model_path = os.environ["FINAL_MODEL_PATH"]
    with open('./ModelConfig.json') as f:
        model_conf = json.load(f)
        logging.info("Model conf: " + str(model_conf))
        epochs = int(model_conf['train']['numTrainEpochs'])
        valid_training_data_ratio = model_conf['train']['validSetRate']
        is_continue_train = model_conf['train']['isContinuous']
        batch_size = 128
        if "MiniBatchs" in model_conf['train']['params']:
            batch_size = model_conf['train']['params']['MiniBatchs']

    learning_rate = model_conf['train']['params']['LearningRate']

    training_data_path = ''
    if "TRAINING_DATA_PATH" in os.environ:
        training_data_path = os.environ["TRAINING_DATA_PATH"]

    # TODO weight_initalizer should be set and enabled

    return {"model_conf": model_conf, "replicas_to_aggregate_ratio": replicas_to_aggregate_ratio, "delimiter": delimiter, 
            "cluster_spec": cluster_spec, "n_pss": n_pss, "n_workers": n_workers, "job_name": job_name, "task_index": task_index, 
            "socket_server_port": socket_server_port, "feature_column_nums": feature_column_nums, 
            "total_training_data_number": total_training_data_number, "sample_weight_column_num": sample_weight_column_num,
            "target_column_num": target_column_num, "tmp_model_path": tmp_model_path, "final_model_path": final_model_path,
            "is_continue_train": is_continue_train, "valid_training_data_ratio": valid_training_data_ratio,
            "layers": model_conf['train']['params']['NumHiddenNodes'], "batch_size": batch_size, "feature_count": feature_count,
            "model_name": model_conf['basic']['name'], "is_chief": is_chief, "training_data_path": training_data_path,
            "export_dir": final_model_path, "epochs": epochs, "sample_weight_column_num": sample_weight_column_num,
            "learning_rate": learning_rate, "loss_func": model_conf['train']['params']['Loss'], "optimizer": "adam",
            "weight_initalizer": "xavier", "act_funcs": model_conf['train']['params']['ActivationFunc']}

def main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    shifu_context = read_context_from_env_and_modelconf();
    logging.info("Shifu context: %s" % str(shifu_context))
    
    # This client is used for sync worker training intermediate information with master
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect(("127.0.0.1", shifu_context["socket_server_port"])) # sync to local one and logic processed in local TaskExecutor

    logging.info("Job info: job_name:%s, task_index:%d." % (shifu_context['job_name'], shifu_context['task_index']))

    ps_hosts = shifu_context['cluster_spec']['ps']
    worker_hosts = shifu_context['cluster_spec']['worker']
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # allows this node know about all other nodes
    if shifu_context['job_name'] == 'ps':  # checks if parameter server
        logging.info("Join cluster as ps role.")
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=shifu_context['task_index'])
        server.join()
    else:  # it must be a worker server
        is_chief = (shifu_context['task_index'] == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=shifu_context['task_index'])
        logging.info("Loading data from worker index = %d" % shifu_context['task_index'])

        if "TRAINING_DATA_PATH" in os.environ:
            logging.info("This is a normal worker.")
            training_data_path = os.environ["TRAINING_DATA_PATH"]
            logging.info("Loading data from path = %s." % str(training_data_path))
        else:
            logging.info("This is a backup worker.")

        if shifu_context['is_chief'] and shifu_context['is_continue_train'] and ( gfile.Exists(os.path.join(shifu_context['final_model_path'], 'saved_model.pb')) or gfile.Exists(os.path.join(shifu_context['final_model_path'], 'saved_model.pbtxt')) ):
            logging.info("Adding model loading hook.")
            tf.reset_default_graph()
            # restore from last checkpoint

            with tf.Session() as session:
                logging.info("loading pb models ...")
                tf.saved_model.loader.load(session, [tag_constants.TRAINING, tag_constants.SERVING], shifu_context['final_model_path'])
                logging.info("loading pb models done and saving to checkpoint ...")
                # global step set to 1000000 to make sure last checkpoint, set to max int ?
                save_path = tf.train.Saver().save(session, shifu_context['tmp_model_path'], global_step=1000000)
                logging.info("loading checkpoint model is done ..." + str(save_path))
            tf.reset_default_graph()

        # import data
        context = load_data(shifu_context)

        # split data into batch
        total_batch = int(len(context["train_data"]) / shifu_context['batch_size'])
        x_batch = np.array_split(context["train_data"], total_batch)
        y_batch = np.array_split(context["train_target"], total_batch)
        sample_w_batch = np.array_split(context["train_data_sample_weight"], total_batch)

        logging.info("Testing set size: %d." % len(context['valid_data']))
        logging.info("Training set size: %d." % len(context['train_data']))

        valid_x = np.asarray(context["valid_data"])
        valid_y = np.asarray(context["valid_target"])
        valid_sample_w = np.asarray(context["valid_data_sample_weight"])

        # Graph
        worker_device = "/job:%s/task:%d" % (shifu_context['job_name'], shifu_context['task_index'])
        with tf.device(tf.train.replica_device_setter(#ps_tasks=n_pss,
                                                      cluster=cluster,
                                                      worker_device=worker_device
                                                      )):
            label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, 1))
            sample_weight_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            keras.backend.set_learning_phase(1)
            keras.backend.manual_variable_initialization(True)
            new_model = model(shifu_context)

            logging.info("Model inputs: %s; Model outputs: %s; Loss: %s; optimizer: %s."  % (str(new_model.inputs), str(new_model.output), str(new_model.loss), str(new_model.optimizer)))


            if new_model.optimizer.__class__.__name__ == "TFOptimizer":
                de_optimizer = new_model.optimizer.optimizer
                logging.info("DEBUG: TFOptimizer.")
            else:
                de_optimizer = get_optimizer(new_model.optimizer.__class__.__name__)

            loss = get_loss_func(new_model.loss)(predictions=new_model.output, labels=label_placeholder, weights=sample_weight_placeholder)

            # Construct SyncReplicasOptimizer for sync model distributed training, async model performance is not good
            batch_size = shifu_context['batch_size']
            valid_ratio = shifu_context['valid_training_data_ratio']
            replicas_to_aggregate_ratio = shifu_context['replicas_to_aggregate_ratio']
            n_training = shifu_context['total_training_data_number']
            opt = tf.train.SyncReplicasOptimizer(
                de_optimizer,
                replicas_to_aggregate=int(n_training * (1-valid_ratio) / batch_size * replicas_to_aggregate_ratio),
                total_num_replicas=int(n_training * (1-valid_ratio) / batch_size),
                name="shifu_sync_replicas")

            global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False,
                                  dtype=tf.int32)

            train_step = opt.minimize(loss, global_step=global_step)
            logging.info("Train step: %s." % str(train_step))
            # init ops
            init_tokens_op = opt.get_init_tokens_op()
            # initialize local step
            local_init = opt.local_step_init_op
            if is_chief:
                # initializes token queue
                local_init = opt.chief_init_op

            # checks if global vars are init
            ready_for_local_init = opt.ready_for_local_init_op

            # Initializing the variables
            init_op = tf.initialize_all_variables()
            logging.info("---Variables initialized---")

        # **************************************************************************************
        # Session
        sync_replicas_hook = opt.make_session_run_hook(is_chief)
        stop_hook = tf.train.StopAtStepHook(num_steps=shifu_context['epochs'])
        chief_hooks = [sync_replicas_hook, stop_hook]
        if shifu_context['is_continue_train']:
            scaff = None
        else:
            scaff = tf.train.Scaffold(init_op=init_op,
                                  local_init_op=local_init,
                                  ready_for_local_init_op=ready_for_local_init)
        # Configure
        if "IS_BACKUP" in os.environ:
            config = tf.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True,
                                    device_filters=['/job:ps', '/job:worker/task:0',
                                                    '/job:worker/task:%d' % shifu_context['task_index']])
        else:
            config = tf.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True)

        # Create a "supervisor", which oversees the training process.
        sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                 is_chief=shifu_context['is_chief'],
                                                 config=config,
                                                 scaffold=scaff,
                                                 hooks=chief_hooks,
                                                 log_step_count_steps=0,
                                                 stop_grace_period_secs=10,
                                                 checkpoint_dir=shifu_context['tmp_model_path'])

        if shifu_context['is_chief'] and not shifu_context['is_continue_train']:
            sess.run(init_tokens_op)
            logging.info("Chief worker start waiting 20 seconds.")
            time.sleep(20)  # grace period to wait on other workers before starting training
            logging.info("Chief worker finish waiting 20 seconds.")

        # Train until hook stops session
        logging.info('Starting training on worker %d.' % shifu_context['task_index'])

        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        while not sess.should_stop():
            try:
                start = time.time()
                for i in range(total_batch):
                    train_feed = {new_model.inputs[0]: x_batch[i],
                                  label_placeholder: y_batch[i],
                                  sample_weight_placeholder: sample_w_batch[i]}

                    _, l, gs = sess.run([train_step, loss, global_step], feed_dict=train_feed, options=run_options,run_metadata=run_metadata)
                training_time = time.time() - start
                
                valid_start = time.time()
                # compute validation loss
                valid_loss, gs = sess.run([loss, global_step], feed_dict={new_model.inputs[0]: valid_x,
                                                                          label_placeholder: valid_y,
                                                                          sample_weight_placeholder: valid_sample_w}
                                          )
                valid_time = time.time() - valid_start
                # TODO, here training loss is last index of loss, should be averaged
                logging.info("DEBUG: total_batch=%s index:%s step: %s worker: %s training loss:%s training time:%s valid loss:%s valid time:%s." % (str(total_batch), str(i), str(gs), str(shifu_context['task_index']), str(l), str(training_time), str(valid_loss), str(valid_time)))

                # Send intermediate result to master
                message = "worker_index:{},time:{},current_epoch:{},training_loss:{},valid_loss:{},valid_time:{}\n".format(
                    str(shifu_context['task_index']), str(training_time), str(gs), str(l), str(valid_loss), str(valid_time))
                if sys.version_info < (3, 0):
                    socket_client.send(bytes(message))
                else:
                    socket_client.send(bytes(message), 'utf8')

            except RuntimeError as re:
                if 'Run called even after should_stop requested.' == re.args[0]:
                    logging.info('About to execute sync_clean_up_op!')
                else:
                    raise

        # close session and log done
        logging.info('Done training task %s.' % str(shifu_context['task_index']))
        sess.close()

        # We just need to make sure chief worker exit with success status is enough
        if shifu_context['is_chief']:
            tf.reset_default_graph()
            # restore from last checkpoint
            with tf.get_default_graph().as_default():
                new_model = model(shifu_context)
                logging.info("Expose model inputs: %s; Model outputs: %s." % (str(new_model.inputs), str(new_model.output)))

            saver = tf.train.Saver()
            with tf.Session() as sess:
                tmp_model_path= shifu_context['tmp_model_path']
                ckpt = tf.train.get_checkpoint_state(tmp_model_path)
                logging.info("Checkpoint path: %s." % ckpt)
                assert ckpt, "Invalid model checkpoint path: {}.".format(tmp_model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

                final_model_path = shifu_context['final_model_path']
                logging.info("Exporting saved_model to: %s." % final_model_path)

                # exported signatures defined in code
                simple_save(session=sess, export_dir=final_model_path,
                            inputs={
                                "shifu_input_0": new_model.inputs[0]
                            },
                            outputs={
                                "shifu_output_0": new_model.output
                            },
                            norm_type=shifu_context['model_conf']['normalize']['normType'])
                logging.info("Export saved_model.")

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            logging.info("DEBUG: ctf: %s " % str(ctf))

            f = tf.gfile.GFile(tmp_model_path + "/timeline.json", mode="w+")
            f.write(ctf)
            time.sleep(20) # grace period to wait before closing session

        logging.info('Session from worker %d closed cleanly.' % shifu_context['task_index'])
        sys.exit()

def load_data(shifu_context):
    valid_ratio = shifu_context['valid_training_data_ratio']
    data_file_list = shifu_context['training_data_path'].split(",")

    logging.info("Input data %s." % data_file_list)
    logging.info("Select columns: %s." % str(shifu_context['feature_column_nums']))

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
    feature_column_nums = shifu_context['feature_column_nums']
    target_column_num = shifu_context['target_column_num']
    sample_weight_column_num = shifu_context['sample_weight_column_num']

    for currentFile in data_file_list:
        logging.info(
            "Now loading %s Progress: %s/%s." % (currentFile, str(file_count), str(len(data_file_list))))
        file_count += 1

        with gfile.Open(currentFile, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            while True:
                line = gf.readline()
                if len(line) == 0:
                    break

                line_count += 1
                if line_count % 10000 == 0:
                    logging.info("Total loading lines: %s." % str(line_count))

                columns = line.split(shifu_context['delimiter'])

                if feature_column_nums is None:
                    feature_column_nums = range(0, len(columns))

                    feature_column_nums.remove(target_column_num)
                    if sample_weight_column_num >= 0:
                        feature_column_nums.remove(sample_weight_column_num)

                if random.random() >= valid_ratio:
                    # Append training data
                    train_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1": # FIXME, some case target is not 0 or 1
                        train_pos_cnt += 1
                    else:
                        train_neg_cnt += 1
                    single_train_data = []
                    for feature_column_num in feature_column_nums:
                        try:
                            f_val = float(columns[feature_column_num].strip('\n'))
                            if math.isnan(f_val):
                                logging.warn("Warning: value is NaN after parsing %s." % columns[feature_column_num].strip('\n'))
                                f_val = 0.0
                            single_train_data.append(f_val)
                        except:
                            single_train_data.append(0.0) # default value 1
                            logging.warn("Could not convert %s to float." % str(columns[feature_column_num].strip('\n')))
                            logging.info("DEBUG: feature_column_num is %s." % str(feature_column_num))
                    train_data.append(single_train_data)

                    weight = float(columns[len(columns)-1].strip('\n'))
                    if weight < 0.0:
                        logging.warn("Warning: weight is below 0, use default 1.0. weight: %s example: %s." % (weight, line))
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
                            f_val = float(columns[feature_column_num].strip('\n'))
                            if math.isnan(f_val):
                                logging.warn("Warning: value is NaN after parsing %s." % columns[feature_column_num].strip('\n'))
                                f_val = 0.0
                            single_valid_data.append(f_val)
                        except:
                            single_valid_data.append(0.0) # default value 1
                            logging.warn("Could not convert %s to float." % str(columns[feature_column_num].strip('\n')))
                            logging.info("DEBUG: feature_column_num is %s." % str(feature_column_num))
                    valid_data.append(single_valid_data)

                    weight = float(columns[len(columns)-1].strip('\n'))
                    if weight < 0.0:
                        logging.warn("Warning: weight is below 0, use default 1.0. weight: %s example: %s." % (weight, line))

                        weight = 1.0
                    valid_data_sample_weight.append([weight])

    logging.info("Total data count: %s." % str(line_count))
    logging.info("Train pos count: %s, neg count: %s." % (str(train_pos_cnt), str(train_neg_cnt)))
    logging.info("Valid pos count: %s, neg count: %s." % (str(valid_pos_cnt), str(valid_neg_cnt)))

    return {"train_data": train_data, "train_target": train_target,
            "valid_data": valid_data, "valid_target": valid_target,
            "train_data_sample_weight": training_data_sample_weight,
            "valid_data_sample_weight": valid_data_sample_weight,
            "feature_count": len(feature_column_nums)}

def simple_save(session, export_dir, inputs, outputs, norm_type, legacy_init_op=None, ):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_def_utils.predict_signature_def(inputs, outputs)
    }
    b = builder.SavedModelBuilder(export_dir)
    b.add_meta_graph_and_variables(
        session,
        tags=[tag_constants.TRAINING, tag_constants.SERVING],
        signature_def_map=signature_def_map,
        assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        legacy_init_op=legacy_init_op,
        clear_devices=True)
    b.save()
    export_generic_config(export_dir=export_dir, input=inputs['shifu_input_0'].name, output=outputs['shifu_output_0'].name, norm_type=norm_type)

def export_generic_config(export_dir, input, output, norm_type):
    config_json_str = ""
    config_json_str += "{\n"
    config_json_str += "    \"inputnames\": [\n"
    config_json_str += "        \"" + input + "\"\n"
    config_json_str += "      ],\n"
    config_json_str += "    \"properties\": {\n"
    config_json_str += "         \"algorithm\": \"tensorflow\",\n"
    config_json_str += "         \"tags\": [\"train\", \"serve\"],\n"
    config_json_str += "         \"outputnames\": \"" + output + "\",\n"
    config_json_str += "         \"normtype\": \"" + norm_type + "\"\n"
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

    logging.info("Starting TensorBoard with --logdir= %s in daemon thread ..." % checkpoint_dir)
    tb_thread.start()

if __name__ == '__main__':
    tf.app.run()