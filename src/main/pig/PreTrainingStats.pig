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
SET mapred.task.timeout 1200000;
SET job.name 'shifu statistic'

DEFINE IsDataFilterOut  ml.shifu.shifu.udf.PurifyDataUDF('$source_type', '$path_model_config', '$path_column_config');
DEFINE AddColumnNum     ml.shifu.shifu.udf.AddColumnNumUDF('$source_type', '$path_model_config', '$path_column_config', 'false');
DEFINE CalculateStats   ml.shifu.shifu.udf.CalculateStatsUDF('$source_type', '$path_model_config', '$path_column_config', 'false');


d = LOAD '$path_raw_data' USING PigStorage('$delimiter');
d = FILTER d BY IsDataFilterOut(*);

d = FOREACH d GENERATE AddColumnNum(*);

d = FOREACH d GENERATE FLATTEN($0);
d = FILTER d BY $0 IS NOT NULL;
d = GROUP d BY $0;


d = FOREACH d {
        t = FOREACH $1 GENERATE $1, $2, $3;
        GENERATE group, t;
}



d = FOREACH d GENERATE FLATTEN(CalculateStats(*));
STORE d INTO '$path_pre_training_stats' USING PigStorage('|', '-schema');