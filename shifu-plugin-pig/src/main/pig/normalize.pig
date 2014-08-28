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


DEFINE Normalize        ml.shifu.plugin.pig.normalization.PigNormalizeUDF('$pathPMML','$modelName');

raw = LOAD '$pathInputData' USING PigStorage('$delimiter');

normalized = FOREACH raw GENERATE Normalize(*);
normalized = FILTER normalized BY $0 IS NOT NULL;
normalized = FOREACH normalized GENERATE FLATTEN($0);

STORE normalized INTO '$pathOutputData' USING PigStorage('|', '-schema');

