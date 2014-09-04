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
package ml.shifu.plugin.spark.stats.unitstates;

import org.dmg.pmml.Counts;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

public class FrequencyUnitState implements UnitState {

    private static final long serialVersionUID = 1L;
    private double totalFreq = 0;
    private double missingFreq = 0;
    private double invalidFreq = 0;

    public FrequencyUnitState() {
        this.totalFreq= 0.0;
        this.missingFreq= 0.0;
        this.invalidFreq= 0.0;
    }
    
    public UnitState getNewBlank() {
        return new FrequencyUnitState();
    }

    public void merge(UnitState state) throws Exception {
        if(!(state instanceof FrequencyUnitState))
            throw new Exception("Expected FrequencyUnitState, got " + state.getClass().toString());
        FrequencyUnitState newState= (FrequencyUnitState) state;
        this.totalFreq+= newState.getTotal();
        this.missingFreq+= newState.getMissing();
        this.invalidFreq+= newState.getInvalid();
    }

    public void addData(Object objValue) {
        this.totalFreq++;
        // check if double, for optimization 
        if(objValue instanceof Double)
            return;
        
        if(objValue==null || objValue.toString().length()==0)
            this.missingFreq++;
        else if(!isNumeric(objValue.toString()))   
            this.invalidFreq++;            
    }

    public double getInvalid() {
        return this.invalidFreq;
    }
    public double getMissing() {
        return this.missingFreq;
    }
    public double getTotal() {
        return this.totalFreq;
    }
    
    
    private boolean isNumeric(String str)  
    {  
        try  {  
            Double.parseDouble(str);  
        }  catch(NumberFormatException e)  {
            return false;  
        }  
        return true;  
    }
    
    public void populateUnivariateStats(UnivariateStats univariateStats, Params params) {
        Counts counts= univariateStats.getCounts();
        if(counts==null) {
            counts= new Counts();
        }
        counts.withInvalidFreq(this.invalidFreq);
        counts.withMissingFreq(this.missingFreq);
        counts.withTotalFreq(this.totalFreq);        
        
        univariateStats.withCounts(counts);
    }
    
}
