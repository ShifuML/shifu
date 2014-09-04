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
package ml.shifu.plugin.spark.stats.interfaces;

import ml.shifu.core.util.Params;

import org.dmg.pmml.UnivariateStats;

/**
 * The most basic aggregator, which 
 * 1. keeps track of a set of values which are always grouped together
 * 2. populates a UnivariateStats object based on its internal state
 */

public interface UnitState extends java.io.Serializable {

    UnitState getNewBlank();
    void merge(UnitState state) throws Exception;
    void addData(Object value);
    public void populateUnivariateStats(UnivariateStats univariateStats, Params params);

}
