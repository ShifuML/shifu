#!/bin/bash
#
# Copyright (c) 2017 PayPal Corporation. All rights reserved.
#
# --------------------------------------------------------------------------------------------------------------
#
#   Instead of directly debug program on local host, currently we have to debug the python program on Horton box wrapper
#   with GLIB environment.
#   This script give you ability to start a python interactive console with well set environment if no parameters is
#   given. Also can running test script if you specific the script file as the first parameter.
#
#   Author: haifwu@paypal.com
#   Create: 1/21/2019
#
# --------------------------------------------------------------------------------------------------------------
# Environment variable
# --------------------------------------------------------------------------------------------------------------
HADOOP_HOME=/usr/hdp/2.6.5.0-292/hadoop
GLIBC_HOME=/x/home/website/glibc2.17
LD_LIBRARY_PATH=/usr/hdp/current/hadoop-client/bin/../lib/native/Linux-amd64-64:/usr/hdp/current/hadoop-client/bin/../lib/native
JAVA_HOME=/usr/java/latest
PYTHON_HOME=/x/home/website/python2.7

source ${HADOOP_HOME}/libexec/hadoop-config.sh
export CLASSPATH=${CLASSPATH}:`${HADOOP_HOME}/bin/hadoop classpath --glob`
export HADOOP_HDFS_HOME="$HADOOP_HOME/../hadoop-hdfs"
# --------------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------------------
function running_python_local(){
    ${GLIBC_HOME}/lib/ld-linux-x86-64.so.2 \
    --library-path ${GLIBC_HOME}/lib:${LD_LIBRARY_PATH}:/lib64:${HADOOP_HOME}/../usr/lib/:${JAVA_HOME}/jre/lib/amd64/server \
    ${PYTHON_HOME}/bin/python
}

function test_script(){
    python_script_name=$1

    ${GLIBC_HOME}/lib/ld-linux-x86-64.so.2 \
    --library-path ${GLIBC_HOME}/lib:${LD_LIBRARY_PATH}:/lib64:${HADOOP_HOME}/../usr/lib/:${JAVA_HOME}/jre/lib/amd64/server \
    ${PYTHON_HOME}/bin/python ${python_script_name}
}

# --------------------------------------------------------------------------------------------------------------
# Shell flow
# --------------------------------------------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    running_python_local
else
    script_name=$1
    test_script ${script_name}
fi