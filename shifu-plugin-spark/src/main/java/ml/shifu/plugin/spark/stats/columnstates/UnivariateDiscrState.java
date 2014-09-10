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
package ml.shifu.plugin.spark.stats.columnstates;

import java.util.ArrayList;

import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
import ml.shifu.plugin.spark.stats.unitstates.HistogramUnitState;

public class SimpleUnivariateDiscrState extends ColumnState {
    
    public SimpleUnivariateDiscrState(String name, Params parameters) {
        params= parameters;
        int maxHistogramSize= Integer.parseInt(params.get("maxHistogramSize", "10000").toString());
        states= new ArrayList<UnitState>();
        states.add(new FrequencyUnitState());
        states.add(new HistogramUnitState(maxHistogramSize));
        fieldName= name;
    }
    
    public ColumnState getNewBlank() {
        return new SimpleUnivariateDiscrState(this.fieldName, this.params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof SimpleUnivariateDiscrState))
            throw new Exception("Expected UnivariateDiscrState in merge, got " + colState.getClass().toString());
    }

}
