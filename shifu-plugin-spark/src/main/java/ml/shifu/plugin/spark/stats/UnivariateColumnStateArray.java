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

import com.google.common.base.Splitter;

import ml.shifu.core.util.CommonUtils;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;
import ml.shifu.plugin.spark.utils.ColType;

/**
 * Implementation of ColumnStateArray for Univariate stats.
 */
public class UnivariateColumnStateArray extends ColumnStateArray {
    
	private static final long serialVersionUID = 1L;
	
    public UnivariateColumnStateArray(String delimiter, List<ColumnState> states) {
        this.delimiter= delimiter;
        this.states= states;
    }

	    
    public void checkClass(ColumnStateArray stateArray) throws Exception {
        if(!(stateArray instanceof UnivariateColumnStateArray))
            throw new Exception("Expected SimpleUnivariateColumnStateArray, got " + stateArray.getClass().toString());

    }

    @Override
    public void addData(String line) {
        int index= 0;
        ColumnState state;
        for(String strValue: Splitter.on(delimiter).split(line)) {
        	state= states.get(index);
        	// if colType is continuous, optimize by sending in parsed Double. If Double is not valid send string for frequency count.
        	if(state.getColType()== ColType.CONTINUOUS) {
    			if(CommonUtils.isValidNumber(strValue))
    				state.addData(Double.valueOf(strValue));
    			else
    				state.addData(strValue);
        	}
        	else
        		state.addData(strValue);
            index++;
        }
        return;
    }

    @Override
    public ColumnStateArray getNewBlank() throws Exception {
    	// copy states
    	List<ColumnState> newStates= new ArrayList<ColumnState>();
    	for(ColumnState state: states)
    		newStates.add(state.getNewBlank());
    	return new UnivariateColumnStateArray(delimiter, newStates);
    }
}
