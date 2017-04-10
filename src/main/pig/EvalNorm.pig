/**
 * Copyright [2012-2014] PayPal Software Foundation
 *  
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *  
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
REGISTER $path_jar;

SET pig.exec.reducers.max 999;
SET pig.exec.reducers.bytes.per.reducer 536870912;
SET mapred.job.queue.name $queue_name;
SET job.name 'Shifu Evaluation: $data_set';
SET mapred.child.java.opts -Xmx1G;
SET mapred.child.ulimit 2.5G;
SET mapred.reduce.slowstart.completed.maps 0.6;
SET mapred.map.tasks.speculative.execution true;
SET mapred.reduce.tasks.speculative.execution true;
SET mapreduce.map.speculative true;
SET mapreduce.reduce.speculative true;
SET pig.splitCombination true;
SET pig.maxCombinedSplitSize 134217728;

DEFINE IsDataFilterOut          ml.shifu.shifu.udf.PurifyDataUDF('$source_type', '$path_model_config', '$path_column_config', '$eval_set_name');
DEFINE EvalNormalize            ml.shifu.shifu.udf.EvalNormUDF('$source_type', '$path_model_config', '$path_column_config', '$eval_set_name', '$scale');

raw = LOAD '$pathEvalRawData' USING PigStorage('$delimiter', '-noschema');
raw = FILTER raw BY IsDataFilterOut(*);

evalNormalize = FOREACH raw GENERATE FLATTEN(EvalNormalize(*));

STORE evalNormalize INTO '$pathEvalNormalized' USING PigStorage('|', '-schema');
