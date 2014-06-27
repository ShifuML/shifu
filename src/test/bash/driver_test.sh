#!/bin/bash

function print_usage() {
	echo "Usage:              "
	echo "	$0 <modelset-name>"
}

function status_check_func() {
	_status=$1
	if [ ${_status} -ne 0 ]; then
		echo "ABORT - Error occur when executing!"
		exit ${_status}
	fi
}

function run_shifu_func {
	_MODELSET=$1;
	_RUNMODE=$2;
	_SOURCE_TYPE=$3;
	
	# create model set
	shifu new ${_MODELSET}
	status_check_func $?
	
	# enter modelset directory
	pushd ${_MODELSET}
	
	# run init
	shifu init
	status_check_func $?
	
	# change runModelStats into akka
	sed -i "/runModeStats/s/akka/${_RUNMODE}/g" ModelConfig.json
	sed -i "/source/s/LOCAL/${_SOURCE_TYPE}/g" ModelConfig.json
	status_check_func $?
	
	# run stats
	shifu stats
	status_check_func $?
	
	# run varselect
	shifu varselect
	status_check_func $?
	
	# run normalize
	shifu normalize
	status_check_func $?
	
	# run train
	shifu train
	status_check_func $?
	
	# run post-train
	shifu posttrain
	status_check_func $?
	
	# create eval set
	shifu eval -new EvalB
	status_check_func $?
	
	# run performance on the eval set
	shifu eval -run EvalA
	status_check_func $?
	
	popd
	
	rm -rf ${_MODELSET}
}

if [ $# -ne 1 ]; then
	print_usage
	exit 1
fi

MODELSET_NAME=$1

if [ -e ${MODELSET_NAME} ]; then
    echo "modelset - ${MODELSET_NAME} already exists"
    exit
fi

run_shifu_func ${MODELSET_NAME} akka local
# run_shifu_func ${MODELSET_NAME} akka hdfs
# run_shifu_func ${MODELSET_NAME} pig local
# run_shifu_func ${MODELSET_NAME} pig hdfs