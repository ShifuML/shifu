/**
 * Copyright [2012-2014] eBay Software Foundation
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
REGISTER '$path_jar'

SET default_parallel $num_parallel
SET mapred.job.queue.name $queue_name;
SET job.name 'shifu evaluation'

DEFINE IsDataFilterOut          ml.shifu.shifu.udf.PurifyDataUDF('$source_type', '$path_model_config', '$path_column_config', '$eval_set_name');
DEFINE EvalScore                ml.shifu.shifu.udf.EvalScoreUDF('$source_type', '$path_model_config', '$path_column_config', '$eval_set_name');
DEFINE Normalize                ml.shifu.shifu.udf.NormalizeUDF('$source_type', '$path_model_config', '$path_column_config');

raw = LOAD '$pathEvalRawData' USING PigStorage('$delimiter');
raw = FILTER raw BY IsDataFilterOut(*);

evalScore = FOREACH raw GENERATE FLATTEN(EvalScore(*));
evalScore = FILTER evalScore BY $0 IS NOT NULL;
-- leverage hadoop sorting, TODO how to set parallel number here
evalScore = ORDER evalScore BY shifu::$columnIndex ASC;

STORE evalScore INTO '$pathEvalScore' USING PigStorage('|', '-schema');
