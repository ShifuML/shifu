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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.Quantile;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.di.builtin.QuantileCalculator;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

public class DoubleRSampleUnitState extends RSampleUnitState<Double> implements UnitState {

    private static final long serialVersionUID = 1L;
    boolean sorted= false;

    public DoubleRSampleUnitState(int maximumSize) {
        super(maximumSize);
    }

    public UnitState getNewBlank() {
        return new DoubleRSampleUnitState(maxSize);
    }

    public void addData(Object value) {
        if(value instanceof Double) {
            addData((Double) value);
            return;
        }
        Double dVal= Double.parseDouble(value.toString());
        super.addSample(dVal);
    }

    public void addData(Double dVal) {
        super.addSample(dVal);
    }
    
    public void sortSamples() {
        if(sorted==false)
            Collections.sort(this.samples);
        sorted= true;
    }
    
    public void populateUnivariateStats(UnivariateStats univariateStats, Params params) {
        this.sortSamples();
        NumericInfo numInfo= univariateStats.getNumericInfo();
        if(numInfo==null)
            numInfo= new NumericInfo();
        
        int numQuantiles= Integer.parseInt(params.get("numQuantiles", "11").toString());        
        numInfo.withQuantiles(getQuantiles(numQuantiles));
        numInfo.withInterQuartileRange(getInterQuantileRange());
        numInfo.withMedian(getMedian());
        univariateStats.withNumericInfo(numInfo);
    }
    
    public List<Quantile> getQuantiles(int num) {
        this.sortSamples();
        QuantileCalculator quantileCalculator = new QuantileCalculator();
        return quantileCalculator.getEvenlySpacedQuantiles(this.samples, num);
    }
    
    public Double getMedian() {
        if(this.samples.size() == 0)
            return null;
        this.sortSamples();
        return this.samples.get(this.samples.size()/2);
    }
    
    public Double getInterQuantileRange() {
        this.sortSamples();
        int sampleSize= this.samples.size();
        if(sampleSize==0)
            return null;
        return this.samples.get((int) Math.floor(sampleSize * 0.75)) - this.samples.get((int) Math.floor(sampleSize * 0.25));
    }


    public void merge(UnitState state) throws Exception {
        // TODO Auto-generated method stub
        if(!(state instanceof DoubleRSampleUnitState))
            throw new Exception("Expected DoubleRSampleUnitState, got " + state.getClass().toString());
        
        super.merge((DoubleRSampleUnitState) state);        
    }
}
