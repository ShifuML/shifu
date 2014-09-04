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

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.SerializedNumericalValueObject;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.BasicNumericInfoUnitState;
import ml.shifu.plugin.spark.stats.unitstates.BinomialRSampleUnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
/**
 * Expects data in the form of a NumericalValueObject.
 */
public class BinomialContState extends ColumnState {
    private static final long serialVersionUID = 1L;

    public BinomialContState(String name, Params parameters) {
        fieldName= name;
        params= parameters;
        int sampleSize= Integer.parseInt(params.get("sampleSize", "100000").toString());
        int numBins= Integer.parseInt(params.get("numBins", "11").toString());
        states= new ArrayList<UnitState>();
        states.add(new BasicNumericInfoUnitState());
        states.add(new FrequencyUnitState());
        states.add(new BinomialRSampleUnitState(sampleSize, numBins));
    }

    @Override
    public ColumnState getNewBlank() {
        return new BinomialContState(fieldName, params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof BinomialContState))
            throw new Exception("Expected BinomialContState, got " + colState.getClass().toString());
    }
    
    public void addData(Object data) {
        try {
        // send only double values to first two states
        SerializedNumericalValueObject nvo= (SerializedNumericalValueObject) data;
        Double value= nvo.getValue();
        states.get(0).addData(value);
        states.get(1).addData(value);

        // pass whole object to sampler
        states.get(2).addData(nvo);
        } catch (NullPointerException | ClassCastException e) {
            System.out.println(e.getMessage());
        }
    }
}
