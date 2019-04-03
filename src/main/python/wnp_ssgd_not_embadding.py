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
import json
import socket
from tensorflow.python.client import timeline
from tensorflow.contrib.layers.python.layers import embedding_ops

HIDDEN_NODES_COUNT = 20
VALID_TRAINING_DATA_RATIO = 0.1

BUILD_MODEL_BY_CONF_ENABLE = True
REPLICAS_TO_AGGREGATE_RATIO = 1

DELIMITER = '|'
BATCH_SIZE = 256

# read from env
cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
n_pss = len(cluster_spec['ps'])  # the number of parameter servers
n_workers = int(os.environ["WORKER_CNT"])  # the number of worker nodes
job_name = os.environ["JOB_NAME"]
task_index = int(os.environ["TASK_ID"])
socket_server_port = int(os.environ["SOCKET_SERVER_PORT"])  # The port of local java socket server listening, to sync worker training intermediate information with master
total_training_data_number = int(os.environ["TOTAL_TRAINING_DATA_NUMBER"]) # total data

numeric_feature_column_nums = [int(s) for s in str(os.environ["SELECTED_NUMERIC_COLUMN_NUMS"]).split(' ')]  # selected numeric column numbers
category_feature_column_nums = [int(s) for s in str(os.environ["SELECTED_CATEGORY_COLUMN_NUMS"]).split(' ')]  # selected category column numbers
NUMERIC_FEATURE_COUNT = len(numeric_feature_column_nums)
CATEGORY_FEATURE_COUNT = len(category_feature_column_nums)

sample_weight_column_num = int(os.environ["WEIGHT_COLUMN_NUM"])  # weight column number, default is -1
target_column_num = int(os.environ["TARGET_COLUMN_NUM"])  # target column number, default is -1

tmp_model_path = os.environ["TMP_MODEL_PATH"]
final_model_path = os.environ["FINAL_MODEL_PATH"]

# This client is used for sync worker training intermediate information with master
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_client.connect(("127.0.0.1", socket_server_port))


''' embedding is Tensor, ids is Tensor or np.array
def embedding_lookup(embedding, ids):
    row_number = tf.range(0, ids.shape[0], 1, tf.int64)
    indices = tf.stack([row_number, ids], axis=1)
    values = tf.ones([ids.shape[0]], tf.float32)
    dense_shape = [ids.shape[0], embedding.shape[0]]
    sparse = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    return tf.sparse_tensor_dense_matmul(sparse, embedding, adjoint_a=True)
'''

# vocabs is each category number of possible categories [3,3] for example.
def wide_model(numeric_input, category_input, vocabs):
    transpose_category_input = tf.transpose(category_input)
    category_sum = None
    # Append embadding category to numeric_sum
    for i in range(0, len(vocabs)):
        embedding = tf.get_variable("wideem" + str(i), [vocabs[i], 8],
                                    initializer=tf.contrib.layers.xavier_initializer()
                                    #partitioner=tf.fixed_size_partitioner(n_pss))
                                    #partitioner=tf.min_max_variable_partitioner(n_pss, 0, 2 << 10)
                                    )
        # Pick one column from category input
        col = tf.gather(transpose_category_input, [i])[0]
        #col = tf.nn.embedding_lookup(transpose_category_input, [i])[0]

        # Same as make [0001]*[w1,w2,w3,w4] = lookup w4
        #embedded_col = embedding_lookup(tf.identity(embedding), col)  # number * embedding output number
        embedded_col = embedding_ops.embedding_lookup_unique(embedding, col)

        if category_sum is None:
            category_sum = embedded_col
        else:
            category_sum = tf.concat([category_sum, embedded_col], 1)

    tf.set_random_seed(1)
    w = tf.get_variable("W", [numeric_input.shape[1] + category_sum.shape[1], 1], initializer=tf.contrib.layers.xavier_initializer())
    wmodel_logits_sum = tf.matmul(tf.concat([numeric_input, category_sum], 1), w)

    return wmodel_logits_sum


def deep_model(numeric_input, category_input, vocabs, hidden1, hidden2, hidden3):
    embedding_output_cnt = 8

    transpose_category_input = tf.transpose(category_input)

    # append emmbadding category input to numeric
    for i in range(0, len(vocabs)):
        embedding = tf.get_variable("deepem" + str(i), [vocabs[i], embedding_output_cnt],
                                    initializer=tf.contrib.layers.xavier_initializer()
                                    #partitioner=tf.fixed_size_partitioner(n_pss))
                                    #partitioner=tf.min_max_variable_partitioner(n_pss, 0, 2 << 10)
                                    )
        # Pick one column from category input
        col = tf.gather(transpose_category_input, [i])[0]
        #col = tf.nn.embedding_lookup(transpose_category_input, [i])[0]

        embedding_category = embedding_ops.embedding_lookup_unique(embedding, col)
        #embedding_category = embedding_lookup(tf.identity(embedding), col)  # batch_size*embedding_output_cnt

        numeric_input = tf.concat([numeric_input, embedding_category], 1)

    # init
    W1 = tf.get_variable("W1", [numeric_input.shape[1], hidden1], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [hidden1], initializer=tf.zeros_initializer())
#    W2 = tf.get_variable("W2", [hidden1, hidden2], initializer=tf.contrib.layers.xavier_initializer())
#    b2 = tf.get_variable("b2", [hidden2], initializer=tf.zeros_initializer())
#    W3 = tf.get_variable("W3", [hidden2, hidden3], initializer=tf.contrib.layers.xavier_initializer())
#    b3 = tf.get_variable("b3", [hidden3], initializer=tf.zeros_initializer())

    # forward
    Z1 = tf.add(tf.matmul(numeric_input, W1), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.tanh(Z1)  # A1 = relu(Z1)
#    Z2 = tf.add(tf.matmul(A1, W2), b2)  # Z2 = np.dot(W2, a1) + b2
#    A2 = tf.nn.tanh(Z2)  # A2 = relu(Z2)
#    Z3 = tf.add(tf.matmul(A2, W3), b3)  # Z3 = np.dot(W3,Z2) + b3
#    A3 = tf.nn.tanh(Z3)

    return A1


def build_w_d(numeric_input, category_input, vocabs):
    #vocabs = [3, 3]  # colum 0 has 3 vocabs and column 1 has 3 vocabs, only used for build graph

    hidden1 = 50
    hidden2 = 20
    hidden3 = 10

    dmodel_logits = deep_model(numeric_input, category_input, vocabs, hidden1, hidden2, hidden3)
    deep_w = tf.get_variable("deep_w", [hidden1, 1], initializer=tf.contrib.layers.xavier_initializer())
    dmodel_logits_sum = tf.matmul(dmodel_logits, deep_w)  # 3 * 1

    wmodel_logits_sum = wide_model(numeric_input, category_input, vocabs)

    # combine wide and deep
    final_b = tf.get_variable("final_b", [1, 1], initializer=tf.zeros_initializer())
    logits = tf.add(tf.add(dmodel_logits_sum, wmodel_logits_sum), final_b)

    output = tf.nn.sigmoid(logits, name="shifu_output_0")
    return output


def model(numeric_input, category_input, y_, sample_weight, model_conf, vocabs):
    logging.info("worker_num:%d" % n_workers)
    logging.info("total_training_data_number:%d" % total_training_data_number)

    y = build_w_d(numeric_input, category_input, vocabs)
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
        tf.train.AdamOptimizer(learning_rate=learning_rate),
        replicas_to_aggregate=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE * REPLICAS_TO_AGGREGATE_RATIO),
        total_num_replicas=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE),
        name="shifu_sync_replicas")
    train_step = opt.minimize(loss, global_step=global_step)

    return opt, train_step, loss, global_step, y


def main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    logging.info("job_name:%s, task_index:%d" % (job_name, task_index))

    ps_hosts = cluster_spec['ps']
    worker_hosts = cluster_spec['worker']
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # allows this node know about all other nodes
    if job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=task_index)
        server.join()
    else:  # it must be a worker server
        is_chief = (task_index == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=task_index)

        logging.info("Loading data from worker index = %d" % task_index)

        if "TRAINING_DATA_PATH" in os.environ:
            logging.info("This is a normal worker..")
            training_data_path = os.environ["TRAINING_DATA_PATH"]
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

        # import data
        context = load_data(training_data_path)

        # split data into batch
        total_batch = int(len(context["numeric_train_data"]) / BATCH_SIZE)
        numeric_x_batch = np.array_split(context["numeric_train_data"], total_batch)
        category_x_batch = np.array_split(context["category_train_data"], total_batch)
        y_batch = np.array_split(context["train_target"], total_batch)
        sample_w_batch = np.array_split(context["train_data_sample_weight"], total_batch)

        numeric_valid_x = np.asarray(context["numeric_valid_data"])
        category_valid_x = np.asarray(context["category_valid_data"])
        valid_y = np.asarray(context["valid_target"])
        valid_sample_w = np.asarray(context["valid_data_sample_weight"])

        logging.info("Testing set size: %d" % len(context["valid_target"]))
        logging.info("Training set size: %d" % len(context["train_target"]))

        # Graph
        worker_device = "/job:%s/task:%d" % (job_name, task_index)
        with tf.device(tf.train.replica_device_setter(#ps_tasks=n_pss,
                                                      cluster=cluster,
                                                      worker_device=worker_device,
                                                      ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(n_pss, tf.contrib.training.byte_size_load_fn)
                                                      )):
            numeric_input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, NUMERIC_FEATURE_COUNT),
                                               name="shifu_numeric_input_0")
            category_input_placeholder = tf.placeholder(dtype=tf.int64, shape=(None, CATEGORY_FEATURE_COUNT),
                                               name="shifu_category_input_0")

            label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, 1))
            sample_weight_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            opt, train_step, loss, global_step, y = model(numeric_input_placeholder,
                                                          category_input_placeholder,
                                                          label_placeholder,
                                                          sample_weight_placeholder,
                                                          model_conf,
                                                          context["vocabs"])

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
        stop_hook = tf.train.StopAtStepHook(num_steps=EPOCH)
        chief_hooks = [sync_replicas_hook, stop_hook]
        if is_continue_train:
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
                                                    '/job:worker/task:%d' % task_index])
        else:
            config = tf.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True)

        # Create a "supervisor", which oversees the training process.
        sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                 is_chief=is_chief,
                                                 config=config,
                                                 scaffold=scaff,
                                                 hooks=chief_hooks,
                                                 stop_grace_period_secs=10,
                                                 checkpoint_dir=tmp_model_path)

        if is_chief and not is_continue_train:
            sess.run(init_tokens_op)
            #start_tensorboard(tmp_model_path)
            logging.info("chief start waiting 40 sec")
            time.sleep(40)  # grace period to wait on other workers before starting training
            logging.info("chief finish waiting 40 sec")

        # Train until hook stops session
        logging.info('Starting training on worker %d' % task_index)
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        while not sess.should_stop():
            try:
                start = time.time()
                for i in range(total_batch):
                    train_feed = {numeric_input_placeholder: numeric_x_batch[i],
                                  category_input_placeholder: category_x_batch[i],
                                  label_placeholder: y_batch[i],
                                  sample_weight_placeholder: sample_w_batch[i]}

                    _, l, gs = sess.run([train_step, loss, global_step], feed_dict=train_feed, options=run_options,run_metadata=run_metadata)
                    logging.info('finish: ' + str(l))
                    if (i%11) == 10 and is_chief:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        logging.info("print ctf")

                        f = tf.gfile.GFile(final_model_path + "/timeline.json", mode="w+")
                        f.write(ctf)
                training_time = time.time() - start

                time.sleep(5)

                valid_loss, gs = sess.run([loss, global_step], feed_dict={numeric_input_placeholder: numeric_valid_x,
                                                                          category_input_placeholder: category_valid_x,
                                                                          label_placeholder: valid_y,
                                                                          sample_weight_placeholder: valid_sample_w}
                                          )
                logging.info('Step: ' + str(gs) + ' worker: ' + str(task_index) + " training loss:" + str(l) + " valid loss:" + str(valid_loss))

                # Send intermediate result to master
                message = "worker_index:{},time:{},current_epoch:{},training_loss:{},valid_loss:{}\n".format(
                    str(task_index), str(training_time), str(gs), str(l), str(valid_loss))
                if sys.version_info < (3, 0):
                    socket_client.send(bytes(message))
                else:
                    socket_client.send(bytes(message), 'utf8')

            except RuntimeError as re:
                if 'Run called even after should_stop requested.' == re.args[0]:
                    logging.info('About to execute sync_clean_up_op!')
                else:
                    raise

        logging.info('Done' + str(task_index))

        # We just need to make sure chief worker exit with success status is enough
        if is_chief:
            tf.reset_default_graph()

            # add placeholders for input images (and optional labels)
            numeric_input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, NUMERIC_FEATURE_COUNT),
                                               name="shifu_numeric_input_0")
            category_input_placeholder = tf.placeholder(dtype=tf.int64, shape=(None, CATEGORY_FEATURE_COUNT),
                                               name="shifu_category_input_0")

            with tf.get_default_graph().as_default():
                prediction = build_w_d(numeric_input_placeholder, category_input_placeholder, context["vocabs"])

            # restore from last checkpoint
            saver = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(tmp_model_path)
                logging.info("ckpt: {}".format(ckpt))
                assert ckpt, "Invalid model checkpoint path: {}".format(tmp_model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

                logging.info("Exporting saved_model to: {}".format(final_model_path))

                # exported signatures defined in code
                simple_save(session=sess, export_dir=final_model_path,
                            inputs={
                                "shifu_numeric_input_0": numeric_input_placeholder,
                                "shifu_category_input_0": category_input_placeholder
                            },
                            outputs={
                                "shifu_output_0": prediction
                            })
                logging.info("Exported saved_model")

            time.sleep(40) # grace period to wait before closing session

        #sess.close()
        logging.info('Session from worker %d closed cleanly' % task_index)
        sys.exit()


def load_data(data_file):
    data_file_list = data_file.split(",")
    global numeric_feature_column_nums
    global category_feature_column_nums

    logging.info("input data %s" % data_file_list)
    logging.info("numeric_feature_column_nums: " + str(numeric_feature_column_nums))
    logging.info("category_feature_column_nums: " + str(category_feature_column_nums))

    numeric_train_data = []
    category_train_data = []
    train_target = []

    numeric_valid_data = []
    category_valid_data = []
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

                if random.random() >= VALID_TRAINING_DATA_RATIO:
                    # Append training data
                    train_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1":
                        train_pos_cnt += 1
                    else:
                        train_neg_cnt += 1

                    single_numeric_train_data = []
                    for numeric_feature_column_num in numeric_feature_column_nums:
                        try:
                            single_numeric_train_data.append(float(columns[numeric_feature_column_num].strip('\n')))
                        except:
                            logging.info("Could not convert " + str(columns[numeric_feature_column_num].strip('\n') + " to float"))
                            logging.info("feature_column_num: " + str(numeric_feature_column_num))
                    numeric_train_data.append(single_numeric_train_data)

                    single_category_train_data = []
                    for category_feature_column_num in category_feature_column_nums:
                        try:
                            single_category_train_data.append(int(float(columns[category_feature_column_num].strip('\n'))))
                        except:
                            logging.info("Could not convert " + str(columns[category_feature_column_num].strip('\n') + " to int"))
                            logging.info("feature_column_num: " + str(category_feature_column_num))
                    category_train_data.append(single_category_train_data)

                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            logging.info("Warning: weight is below 0. example:" + line)
                            weight = 1.0
                        training_data_sample_weight.append([weight])
                    else:
                        training_data_sample_weight.append([1.0])
                else:
                    # Append validation data
                    valid_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1":
                        valid_pos_cnt += 1
                    else:
                        valid_neg_cnt += 1

                    single_numeric_valid_data = []
                    for numeric_feature_column_num in numeric_feature_column_nums:
                        try:
                            single_numeric_valid_data.append(float(columns[numeric_feature_column_num].strip('\n')))
                        except:
                            logging.info("Could not convert " + str(columns[numeric_feature_column_num].strip('\n') + " to float"))
                            logging.info("feature_column_num: " + str(numeric_feature_column_num))
                    numeric_valid_data.append(single_numeric_valid_data)

                    single_category_valid_data = []
                    for category_feature_column_num in category_feature_column_nums:
                        try:
                            single_category_valid_data.append(int(float(columns[category_feature_column_num].strip('\n'))))
                        except:
                            logging.info("Could not convert " + str(columns[category_feature_column_num].strip('\n') + " to int"))
                            logging.info("feature_column_num: " + str(category_feature_column_num))
                    category_valid_data.append(single_category_valid_data)

                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            logging.info("Warning: weight is below 0. example:" + line)
                            weight = 1.0
                        valid_data_sample_weight.append([weight])
                    else:
                        valid_data_sample_weight.append([1.0])

    logging.info("Total data count: " + str(line_count) + ".")
    logging.info("Train pos count: " + str(train_pos_cnt) + ", neg count: " + str(train_neg_cnt) + ".")
    logging.info("Valid pos count: " + str(valid_pos_cnt) + ", neg count: " + str(valid_neg_cnt) + ".")

    category_cnt = []
    with open('./ColumnConfig.json') as f:
        column_conf = json.load(f)
        for i in category_feature_column_nums:
            category_cnt.append(len(column_conf[i]["columnBinning"]["binCategory"]) + 1)
    logging.info("category_cnt: " + str(category_cnt) +  ".")

    return {"numeric_train_data": numeric_train_data, "category_train_data": category_train_data, "train_target": train_target,
            "numeric_valid_data": numeric_valid_data, "category_valid_data": category_valid_data, "valid_target": valid_target,
            "train_data_sample_weight": training_data_sample_weight,
            "valid_data_sample_weight": valid_data_sample_weight,
            "feature_count": len(numeric_feature_column_nums) + len(category_feature_column_nums),
            "vocabs": category_cnt}


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


def export_generic_config(export_dir):
    config_json_str = ""
    config_json_str += "{\n"
    config_json_str += "    \"inputnames\": [\n"
    config_json_str += "        \"shifu_numeric_input_0,\"\n"
    config_json_str += "        \"shifu_category_input_0\"\n"
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


if __name__ == '__main__':
    tf.app.run()
