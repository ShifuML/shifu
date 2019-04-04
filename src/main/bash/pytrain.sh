#!/usr/bin/env bash

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
# Enable python based tensorflow training, all training logic redicted to scripts/train.py.
# 

HADOOP_HOME=$1
GLIBC_HOME=$2
LD_LIBRARY_PATH=$3
JAVA_HOME=$4
PYTHON_HOME=$5
train_script=$6
shift
shift
shift
shift
shift
shift

source $HADOOP_HOME/libexec/hadoop-config.sh
export CLASSPATH=$CLASSPATH:`$HADOOP_HOME/bin/hadoop classpath --glob`
export HADOOP_HDFS_HOME="$HADOOP_HOME/../hadoop-hdfs"

echo $@

$GLIBC_HOME/lib/ld-linux-x86-64.so.2 \
    --library-path $GLIBC_HOME/lib:$LD_LIBRARY_PATH:/lib64:$HADOOP_HOME/../usr/lib/:$JAVA_HOME/jre/lib/amd64/server \
    $PYTHON_HOME/bin/python \
    $train_script $@

