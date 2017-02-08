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

-- This is a tool to calculate the PSI from another dataset
SET job.name 'Shifu PSI calculator: $data_set';
SET mapred.map.tasks.speculative.execution true;
SET mapred.reduce.tasks.speculative.execution true;
SET mapreduce.map.speculative true;
SET mapreduce.reduce.speculative true;
SET mapred.job.queue.name $queue_name;

REGISTER $path_jar;

DEFINE AddColumnNum      ml.shifu.shifu.udf.AddColumnNumUDF('$source_type', '$path_model_config', '$path_column_config', 'false');
DEFINE PopulationCounter ml.shifu.shifu.udf.PopulationCounterUDF('$source_type', '$path_model_config', '$path_column_config', '1');
DEFINE PSI               ml.shifu.shifu.udf.PSICalculatorUDF('$source_type', '$path_model_config', '$path_column_config');

data = load 'path_to_data' USING PigStorage('$delimiter');

data_cols = FOREACH data GENERATE AddColumnNum(*);
data_cols = FILTER data_cols by $0 is not null;

population_info = foreach (group data_cols by $0) generate FLATTEN(PopulationCounter(*));

psi = foreach (group population_info by $0) generate FLATTEN(PSI(*));

dump psi;

