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
REGISTER '$path_jar'

SET pig.exec.reducers.max 999;
SET pig.exec.reducers.bytes.per.reducer 536870912;
SET mapred.map.tasks.speculative.execution true;
SET mapred.reduce.tasks.speculative.execution true;
SET mapreduce.map.speculative true;
SET mapreduce.reduce.speculative true;
SET mapred.job.queue.name $queue_name;
SET mapred.task.timeout 1200000;
SET job.name 'Shifu Statistic: $data_set';
SET io.sort.mb 500;
SET mapred.child.java.opts '-Xmx1G -server -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=70';
SET mapred.child.ulimit 2.5G;
SET mapred.reduce.slowstart.completed.maps 0.8;

-- for new hadoop >= 2.4.0
--SET mapreduce.map.java.opts  '-server -Xms2048m -Xmx3072m -Djava.net.preferIPv4Stack=true  -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=70'
--SET mapreduce.reduce.java.opts '-server -Xms1024m -Xmx2048m -Djava.net.preferIPv4Stack=true  -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=70'
--SET mapreduce.task.io.sort.mb 1536
SET mapreduce.reduce.speculative true
SET mapreduce.map.speculative true
SET mapreduce.map.sort.spill.percent 0.95
SET mapreduce.job.reduce.slowstart.completedmaps 0.8

DEFINE IsDataFilterOut  ml.shifu.shifu.udf.PurifyDataUDF('$source_type', '$path_model_config', '$path_column_config');
--DEFINE IsToBinningData  ml.shifu.shifu.udf.FilterBinningDataUDF('$source_type', '$path_model_config', '$path_column_config');
DEFINE GenBinningData   ml.shifu.shifu.udf.BinningPartialDataUDF('$source_type', '$path_model_config', '$path_column_config', '$histo_scale_factor');
DEFINE MergeBinningData ml.shifu.shifu.udf.BinningDataMergeUDF('$source_type', '$path_model_config', '$path_column_config');
DEFINE AddColumnNum     ml.shifu.shifu.udf.AddColumnNumAndFilterUDF('$source_type', '$path_model_config', '$path_column_config', 'false');

-- load and purify data
data = LOAD '$path_raw_data' USING PigStorage('$delimiter', '-noschema');
data = FILTER data BY IsDataFilterOut(*);

-- convert data into column based
data_cols = FOREACH data GENERATE AddColumnNum(*);
data_cols = FILTER data_cols BY $0 IS NOT NULL;
data_cols = FOREACH data_cols GENERATE FLATTEN($0);

-- prepare data and do binning
data_binning_grp = GROUP data_cols BY ($0, $3) PARALLEL $column_parallel;
binning_info_partial = FOREACH data_binning_grp GENERATE group.$0, GenBinningData(data_cols);
binning_info_partial = FILTER binning_info_partial BY $1 IS NOT NULL;
binning_info_grp = GROUP binning_info_partial BY $0 PARALLEL $group_binning_parallel;
binning_info = FOREACH binning_info_grp GENERATE FLATTEN(MergeBinningData(*));
STORE binning_info INTO '$path_stats_binning_info' USING PigStorage('|', '-schema');
