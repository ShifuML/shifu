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

import java.util.List;

import ml.shifu.core.util.Params;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.UnivariateStats;

/**
 * This class holds a collection of unit states to be maintained for a particular column in the table.
 * ColumnState has instantiations which should define the UnitStates they need in the constructor.
 * 
 */

public abstract class ColumnState implements java.io.Serializable {

    private static final long serialVersionUID = 1L;
    protected List<UnitState> states;
    protected Params params;
    protected String fieldName;

    abstract public ColumnState getNewBlank();
    
    public void merge(ColumnState colState) throws Exception {
        checkClass(colState);
        int index= 0;
        for(UnitState state: colState.getStates()) {
            this.states.get(index).merge(state);
            index++;
        }
    }
        
    abstract public void checkClass(ColumnState colState) throws Exception;
    
    public void addData(Object value) {
        try {
            for(UnitState state: this.states) 
                state.addData(value);
        } catch (NullPointerException | ClassCastException e) {
            System.out.println(e.getMessage());
        }
    }

    public List<UnitState> getStates() {
        return this.states;
    }

    public UnivariateStats getUnivariateStats() {
        
        UnivariateStats univariateStats= new UnivariateStats();
        for(UnitState state:this.states)
            state.populateUnivariateStats(univariateStats, this.params);
        univariateStats.setField(new FieldName(fieldName));
        return univariateStats;
    }
    
}
