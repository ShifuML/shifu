#!/bin/bash

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
# Enable python based tensorflow training, all training logic redicted to python file under ./scripts
# 

echo $HADOOP_HOME
echo $GLIBC_HOME
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native/Linux-amd64-64:$HADOOP_HOME/lib/native"
echo $PYTHON_HOME
echo $TRAIN_SCRIPT_PATH
echo $HADOOP_HOME
echo $LD_LIBRARY_PATH

chmod 777 -R ./

source $HADOOP_HOME/libexec/hadoop-config.sh
export CLASSPATH=$CLASSPATH:`$HADOOP_HOME/bin/hadoop classpath --glob`
export HADOOP_HDFS_HOME="$HADOOP_HOME/../hadoop-hdfs"

$GLIBC_HOME/lib/ld-linux-x86-64.so.2 \
    --library-path $GLIBC_HOME/lib:$LD_LIBRARY_PATH:/lib64:$HADOOP_HOME/../usr/lib/:$JAVA_HOME/jre/lib/amd64/server \
    $PYTHON_HOME/bin/python \
    $TRAIN_SCRIPT_PATH $@