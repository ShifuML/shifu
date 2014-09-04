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

import org.dmg.pmml.ModelStats;

import com.google.common.base.Splitter;

/**
 * This class holds an array of states for every column in the table.
 * Two instances exist for Univariate and Binomial Stats.
 */

public abstract class ColumnStateArray implements java.io.Serializable {

    private static final long serialVersionUID = 1L;
    protected List<ColumnState> states;
    protected String delimiter;
        
    abstract public ColumnStateArray getNewBlank() throws Exception;
    
    public ColumnStateArray merge(ColumnStateArray stateArray2) throws Exception {
        checkClass(stateArray2);
        if(stateArray2.getStateArray().size() != states.size())
            throw new Exception("Sizes of state arrays don't match");
        int index= 0;
        for(ColumnState colState: stateArray2.getStateArray()) {
            states.get(index).merge(colState);
            index++;
        }
        return this;
    }
    
    public abstract void checkClass(ColumnStateArray stateArray) throws Exception;

    public void addData(String line) {
        int index= 0;
        for(String strValue: Splitter.on(delimiter).split(line)) {
            states.get(index).addData(strValue);
            index++;
        }
        return;
    }
    
    public List<ColumnState> getStateArray() {
        return states;
    }

    public ModelStats getModelStats() {
        ModelStats modelStats= new ModelStats();
        for(ColumnState state: this.states) {
            modelStats.withUnivariateStats(state.getUnivariateStats());
        }
        
        return modelStats;
    }

}
