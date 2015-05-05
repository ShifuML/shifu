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
REGISTER $path_jar;

SET default_parallel $num_parallel;
SET mapred.job.queue.name $queue_name;
SET job.name 'shifu normalize';
SET io.sort.mb 500;
SET mapred.child.java.opts -Xmx1G;
SET mapred.child.ulimit 2.5G;
-- to disable compress output to make sure in train step, input files can be merged into one worker, this is a 
-- work-around solution to solve issue on train.
SET mapred.output.compress false;

DEFINE IsDataFilterOut  ml.shifu.shifu.udf.PurifyDataUDF('$source_type', '$path_model_config', '$path_column_config');
DEFINE Normalize        ml.shifu.shifu.udf.NormalizeUDF('$source_type', '$path_model_config', '$path_column_config');

raw = LOAD '$path_raw_data' USING PigStorage('$delimiter');
filtered = FILTER raw BY IsDataFilterOut(*);

STORE filtered INTO '$pathSelectedRawData' USING PigStorage('$delimiter', '-schema');

normalized = FOREACH filtered GENERATE Normalize(*);
normalized = FILTER normalized BY $0 IS NOT NULL;
normalized = FOREACH normalized GENERATE FLATTEN($0);

STORE normalized INTO '$pathNormalizedData' USING PigStorage('|', '-schema');
