#!/bin/bash

LAST_N_JOB_KILL=5

echo "Try to kill last#${LAST_N_JOB_KILL} mapreduce or spark jobs in Horton"
grep -o "application_[0-9_]*" logs/shifu.log | uniq | tail -${LAST_N_JOB_KILL} | while read application_id
do
    echo "kill ${application_id} ... "
    yarn application -kill ${application_id}
done