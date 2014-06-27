# /bin/bash

# Copyright [2012-2014] eBay Software Foundation
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

############################################################################################
# Bash Competion for core command
# To use it:
#  1. Set core command to PATH env:
#     export PATH=${PATH}:${SHIFU_HOME}/bin/
#  2. Source this file:
#     source shifu_bash_completion.sh
#  3. Enjoy it:
#     core <tab>
############################################################################################

function _shifu_completion() {
    COMPREPLY=()
    local cur=${COMP_WORDS[COMP_CWORD]};
    local com=${COMP_WORDS[COMP_CWORD-1]};
    case $com in
    'core')
        COMPREPLY=($(compgen -W 'new cp init stats varselect normalize train posttrain eval' -- $cur))
        ;;
    'eval')
        COMPREPLY=($(compgen -W '-new -run -perf' -- $cur))
        ;;
    'train')
        COMPREPLY=($(compgen -W '-dry -yarn' -- $cur))
        ;;
    *)
        ;;
    esac
    return 0
}

complete -F _shifu_completion shifu
