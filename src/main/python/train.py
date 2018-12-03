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

#
# Python Tensorflow NN Training, user can change TF DAG to customize NN used for training. Models will be saved in the
# same folder of regular models in 'models' folder and being evaluated in distributed shifu eval step.
#

import gzip
from StringIO import StringIO

from tensorflow.python.platform import gfile

print("Hello World!")
print(gfile.ListDirectory("hdfs://horton/user/pengzhang/ModelSets/demo/tmp/NormalizedData/"))

with gfile.Open("hdfs://horton/user/pengzhang/10.txt", 'rb') as rf:
    print(rf.readline())

with gfile.Open("hdfs://horton/user/pengzhang/ModelSets/demo/tmp/NormalizedData/part-m-00000.gz", 'rb') as rf:
    print(gzip.GzipFile(fileobj=StringIO(rf.read())).readline())
