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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataField;
import org.dmg.pmml.OpType;

import com.google.common.base.Splitter;

import ml.shifu.core.container.CategoricalValueObject;
import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.columnstates.BinomialContState;
import ml.shifu.plugin.spark.stats.columnstates.BinomialDiscrState;
import ml.shifu.plugin.spark.stats.columnstates.BinomialOrdinalState;
import ml.shifu.plugin.spark.stats.columnstates.SimpleUnivariateOrdinalState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;

/**
 * Implementation of ColumnStateArray for Binomial Stats. 
 */

public class BinomialColumnStateArray extends ColumnStateArray {

    private enum ColType {
        ORDINAL, CATEGORICAL, CONTINUOUS
    }
    
    private static final long serialVersionUID = 1L;
    final int targetFieldNum;
    Set<String> posTags;
    Set<String> negTags;
    List<ColType> columnTypes;
    
    public BinomialColumnStateArray(BinomialColumnStateArray initValue) {
        // creates a blank copy of stateArray
        states= new ArrayList<ColumnState>();
        for(ColumnState s: initValue.getStateArray()) {
            states.add(s.getNewBlank());
        }
        delimiter= initValue.delimiter;
        targetFieldNum= initValue.targetFieldNum;
        columnTypes= initValue.getColumnTypes();
        posTags= initValue.getPosTags();
        negTags= initValue.getNegTags();
    }
    


    // TODO: add weights functionality
    public BinomialColumnStateArray(List<DataField> dataFields, Params params) {
        delimiter= params.get("delimiter", ",").toString();
        posTags= new HashSet<String>((List<String>) params.get("posTags"));
        negTags= new HashSet<String>((List<String>) params.get("negTags"));
        targetFieldNum= Integer.parseInt(params.get("targetFieldNum").toString());
        states= new ArrayList<ColumnState>();
        columnTypes= new ArrayList<ColType>();

        // create a SimpleUnivariateOrdinal column state for the target field and ordinal fields
        int index=0;
        for(DataField field: dataFields) {
            if(field.getOptype().equals(OpType.ORDINAL)){
                states.add(new SimpleUnivariateOrdinalState(field.getName().getValue(), params));                
                columnTypes.add(ColType.ORDINAL);
            }
            else if(field.getOptype().equals(OpType.CATEGORICAL)) {
                states.add(new BinomialDiscrState(field.getName().getValue(), params));
                columnTypes.add(ColType.CATEGORICAL);
            }
            else if(field.getOptype().equals(OpType.CONTINUOUS)) {
                states.add(new BinomialContState(field.getName().getValue(), params));
                columnTypes.add(ColType.CONTINUOUS);                
            }
            index++;
        }   
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
            if(columnTypes.get(index)==ColType.CATEGORICAL) {
                // form a CategoricalValueObject
                SerializedCategoricalValueObject cvo= new SerializedCategoricalValueObject();
                cvo.setIsPositive(isPositive);
                cvo.setValue(value);
                cvo.setWeight(1.0);
                // 1 for now, TODO: change
                states.get(index).addData(cvo);
            }
            else if(columnTypes.get(index)==ColType.CONTINUOUS) {
                // form a NumericalValueObject
                if(CommonUtils.isValidNumber(value)) {
                    SerializedNumericalValueObject nvo= new SerializedNumericalValueObject();
                    nvo.setIsPositive(isPositive);
                    nvo.setValue(Double.parseDouble(value));
                    nvo.setWeight(1.0);
                    // 1 for now, TODO: change
                    states.get(index).addData(nvo);
                }
                else {
                    System.out.println("Invalid number found in continuous column: " + value);
                }
            }
            else if(columnTypes.get(index)==ColType.ORDINAL) {
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
        return new BinomialColumnStateArray(this);
    }

    private List<ColType> getColumnTypes() {
        return columnTypes;
    }
    private Set<String> getNegTags() {
        return negTags;
    }


    private Set<String> getPosTags() {
        return posTags;
    }


}
