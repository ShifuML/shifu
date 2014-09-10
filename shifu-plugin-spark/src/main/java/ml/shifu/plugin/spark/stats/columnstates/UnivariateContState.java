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
import java.util.List;

import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.BasicNumericInfoUnitState;
import ml.shifu.plugin.spark.stats.unitstates.DoubleRSampleUnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
import ml.shifu.plugin.spark.stats.unitstates.RSampleUnitState;


public class SimpleUnivariateContState extends ColumnState {
    
    public SimpleUnivariateContState(String name, Params parameters) {
        params= parameters;
        int sampleSize= Integer.parseInt(params.get("sampleSize", "10000").toString());
        states= new ArrayList<UnitState>();
        states.add(new BasicNumericInfoUnitState());
        states.add(new FrequencyUnitState());
        //states.add(new DoubleRSampleUnitState(sampleSize));
        fieldName= name;
    }
    
    public ColumnState getNewBlank() {
        return new SimpleUnivariateContState(fieldName, params);
    }
    
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof SimpleUnivariateContState))
            throw new Exception("Expected UnivariateContState in merge, got " + colState.getClass().toString());
    }
    
    public void addData(Object value) {
        // parse into a double and pass, for optimization
        // TODO: invalid frequencies won't get recorded
        try {
            if(CommonUtils.isValidNumber(value)) {
                Double dVal= Double.valueOf(value.toString());
                for(UnitState state: states)
                    state.addData(dVal);
            }        
        } catch (NullPointerException e) {
            System.out.println(e.getMessage());
        }
    }
}
