#!/bin/bash
grep -B4 '"finalSelect" : true' $1 | grep columnName | awk '{print $NF}' | sed 's/[",]//g'