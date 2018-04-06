#!/usr/bin/env bash
function getProperty () {
    PROP_NAME=$1
    CFG=$2
    PROP_VAL=$(sed -n "/^${PROP_NAME}[ ]*=/p" $CFG | awk -F'=' '{print $2}' | sed -e 's/^[[:space:]]//g' -e 's/[[:space:]]$//g')
    echo ${PROP_VAL}
}

statistics_dir=$(getProperty "shifu.usage.statistics.dir" ${SHIFU_HOME}/conf/shifuconfig)
if [ "${statistics_dir}" == "" ]; then
    statistics_dir=/tmp/shifu
    hdfs dfs -mkdir -p ${statistics_dir} >& /dev/null
    hdfs dfs -chmod 777 ${statistics_dir} >& /dev/null
fi

log_file="shifu-${USER}-$(date +%s).txt"
echo "${USER}@$(hostname):${PWD} > shifu $@" > /tmp/${log_file}
hdfs dfs -put /tmp/${log_file} ${statistics_dir} >& /dev/null