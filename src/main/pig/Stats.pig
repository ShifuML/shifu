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
DEFINE AddColumnNum     ml.shifu.shifu.udf.AddColumnNumUDF('$source_type', '$path_model_config', '$path_column_config');
DEFINE BINNING          ml.shifu.shifu.udf.StreamingBinningUDF('$source_type', '$path_model_config', '$path_column_config');
DEFINE Calculator       ml.shifu.shifu.udf.StreamingCalculateStatesUDF('$source_type', '$path_model_config', '$path_column_config');

--data = load '$data_input' using PigStorage($delimiter);

data = LOAD '$path_raw_data' USING PigStorage('$delimiter');
data = FILTER data BY IsDataFilterOut(*);

data = FOREACH data GENERATE AddColumnNum(*);

data = FOREACH data GENERATE FLATTEN($0);
data = FILTER data BY $0 IS NOT NULL;

--step 1, process the binning
s = foreach (group data by $0) generate group as col,
                                        BINNING(data) as binning;

--step 2, based on the binning, process other metrics
d = foreach (group s by col, data by $0) generate group as col,
                                                  FLATTEN(d.binning) as binning,
                                                  FLATTEN(Calculator(d.binning, data));

store d into 'result' using PigStorage('|');



