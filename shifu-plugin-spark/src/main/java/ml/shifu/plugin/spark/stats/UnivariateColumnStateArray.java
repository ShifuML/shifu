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
package ml.shifu.plugin.spark.stats;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.columnstates.SimpleUnivariateContState;
import ml.shifu.plugin.spark.stats.columnstates.SimpleUnivariateDiscrState;
import ml.shifu.plugin.spark.stats.columnstates.SimpleUnivariateOrdinalState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;

import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;
/**
 * Implementation of ColumnStateArray for Univariate stats.
 */
public class UnivariateColumnStateArray extends ColumnStateArray {
    
    public UnivariateColumnStateArray(UnivariateColumnStateArray initValue) {
        // creates a blank copy of stateArray
        states= new ArrayList<ColumnState>();
        for(ColumnState s: initValue.getStateArray()) {
            states.add(s.getNewBlank());
        }
        delimiter= initValue.delimiter;
    }
    
    public UnivariateColumnStateArray(List<DataField> dataFields, Params params) {
        states= new ArrayList<ColumnState>();
        for(DataField field: dataFields) {
            if(field.getOptype().equals(OpType.CATEGORICAL))
                states.add(new SimpleUnivariateDiscrState(field.getName().getValue(), params));
            else if(field.getOptype().equals(OpType.CONTINUOUS))
                states.add(new SimpleUnivariateContState(field.getName().getValue(), params));
            else if(field.getOptype().equals(OpType.ORDINAL))
                states.add(new SimpleUnivariateOrdinalState(field.getName().getValue(), params));
        }
        delimiter= params.get("delimiter", ",").toString();
    }

    public void checkClass(ColumnStateArray stateArray) throws Exception {
        if(!(stateArray instanceof UnivariateColumnStateArray))
            throw new Exception("Expected SimpleUnivariateColumnStateArray, got " + stateArray.getClass().toString());

    }
    
    @Override
    public ColumnStateArray getNewBlank() throws Exception {
        return new UnivariateColumnStateArray(this);
    }
}
