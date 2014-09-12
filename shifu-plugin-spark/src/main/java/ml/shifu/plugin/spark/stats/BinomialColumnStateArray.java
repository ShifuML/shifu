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
import java.util.Set;

import com.google.common.base.Splitter;

import ml.shifu.core.util.CommonUtils;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;
import ml.shifu.plugin.spark.utils.ColType;

/**
 * Implementation of ColumnStateArray for Binomial Stats. 
 */

public class BinomialColumnStateArray extends ColumnStateArray {

    private static final long serialVersionUID = 1L;
    final int targetFieldNum;
    Set<String> posTags;
    Set<String> negTags;
    
    public BinomialColumnStateArray(String delimiter, Set<String> posTags, Set<String> negTags, int targetFieldNum, List<ColumnState> states) {
    	this.delimiter= delimiter;
    	this.posTags= posTags;
    	this.negTags= negTags;
    	this.states= states;
    	this.targetFieldNum= targetFieldNum;
    }

    @Override
    public void addData(String line) {
        // find the tag from line and create BinomialTuples
        List<String> parsedLine= new ArrayList<String>();
        for(String strValue: Splitter.on(delimiter).split(line))
            parsedLine.add(strValue);
        String tag= parsedLine.get(targetFieldNum);
        boolean isPositive= posTags.contains(tag);
        // check for negative tags?
        
        int index= 0;        
        for(String value: parsedLine) {
            if(states.get(index).getColType()==ColType.CATEGORICAL) {
                // form a CategoricalValueObject
                SerializedCategoricalValueObject cvo= new SerializedCategoricalValueObject();
                cvo.setIsPositive(isPositive);
                cvo.setValue(value);
                cvo.setWeight(1.0);
                states.get(index).addData(cvo);
            }
            else if(states.get(index).getColType()==ColType.CONTINUOUS) {
                // form a NumericalValueObject
                if(CommonUtils.isValidNumber(value)) {
                    SerializedNumericalValueObject nvo= new SerializedNumericalValueObject();
                    nvo.setIsPositive(isPositive);
                    nvo.setValue(Double.parseDouble(value));
                    nvo.setWeight(1.0);
                    states.get(index).addData(nvo);
                }
                else {
                    System.out.println("Invalid number found in continuous column: " + value);
                }
            }
            else if(states.get(index).getColType()==ColType.ORDINAL) {
                // pass data as is
                states.get(index).addData(value);
            }
            index++;
        }
    }
    
    @Override
    public void checkClass(ColumnStateArray stateArray) throws Exception {
        if(!(stateArray instanceof BinomialColumnStateArray))
            throw new Exception("Expected BinomialColumnStateArray, got " + stateArray.getClass().toString());
    }
    
    @Override
    public ColumnStateArray getNewBlank() throws Exception {
        List<ColumnState> newStates= new ArrayList<ColumnState>();
        for(ColumnState state: states) 
            newStates.add(state.getNewBlank());

        return new BinomialColumnStateArray(delimiter, posTags, negTags, targetFieldNum, newStates);
    }

}
