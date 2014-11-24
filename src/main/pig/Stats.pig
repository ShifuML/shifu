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
register '$path_jar'

set default_parallel $num_parallel
set mapred.job.queue.name $queue_name;
set mapred.task.timeout 1200000;
set io.sort.mb 500;

set job.name 'shifu statistic'

define IsDataFilterOut  ml.shifu.shifu.udf.PurifyDataUDF('$source_type', '$path_model_config', '$path_column_config');
define AddColumnNum     ml.shifu.shifu.udf.AddColumnNumUDF('$source_type', '$path_model_config', '$path_column_config');
define BINNING          ml.shifu.shifu.udf.StreamingBinningUDF('$source_type', '$path_model_config', '$path_column_config');
define Calculator       ml.shifu.shifu.udf.StreamingCalculateStatesUDF('$source_type', '$path_model_config', '$path_column_config');

--data = load '$data_input' using PigStorage($delimiter);

data = load '$path_raw_data' using PigStorage('$delimiter');
data = filter data by IsDataFilterOut(*);

data = foreach data generate AddColumnNum(*);

data = foreach data generate FLATTEN($0);
data = filter data by $0 is not null and $1 is not null;

--step 1, process the binning
bins = foreach (group data by $0) generate group as col,
                                           BINNING(data) as binning;

--step 2, based on the binning, process other metrics
group_data = group data by $0;
join_data = join group_data by group, bins by col using 'replicated';

-- $0:group, $1:data, $2:col, $3:binning
d = foreach join_data generate col,
                               FLATTEN(Calculator($3, $1));



store d into '$path_pre_training_stats' using PigStorage('|');



