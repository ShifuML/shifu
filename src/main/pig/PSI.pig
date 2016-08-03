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
SET job.name 'shifu PSI calculator';
SET mapred.map.tasks.speculative.execution true;
SET mapred.reduce.tasks.speculative.execution true;
SET mapred.job.queue.name $queue_name;

REGISTER $path_jar;

DEFINE AddColumnNum      ml.shifu.shifu.udf.AddColumnNumUDF('$source_type', '$path_model_config', '$path_column_config', 'false');
DEFINE PopulationCounter ml.shifu.shifu.udf.PopulationCounterUDF('$source_type', '$path_model_config', '$path_column_config', '$value_index');
DEFINE PSI               ml.shifu.shifu.udf.PSICalculatorUDF('$source_type', '$path_model_config', '$path_column_config');

data = LOAD '$path_raw_data' USING PigStorage('$delimiter');

-- not need to filtering
data_cols = FOREACH data GENERATE $PSIColumn, AddColumnNum(*);
data_cols = FILTER data_cols by $1 is not null;
data_cols = FOREACH data_cols GENERATE $PSIColumn, FLATTEN($1);

-- calculate counting number for each column and each psi bin
population_info = foreach (group data_cols by ($PSIColumn, $1) PARALLEL $column_parallel) generate FLATTEN(PopulationCounter(*));

-- calculate the psi
psi = foreach (group population_info by $0) generate FLATTEN(PSI(*));

rmf $path_psi
store psi INTO '$path_psi' USING PigStorage('|', '-schema');


