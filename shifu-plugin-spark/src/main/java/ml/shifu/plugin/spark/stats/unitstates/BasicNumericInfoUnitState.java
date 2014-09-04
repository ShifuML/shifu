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

import org.dmg.pmml.ContStats;
import org.dmg.pmml.Counts;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

public class BasicNumericInfoUnitState implements UnitState {
    private static final long serialVersionUID = 1L;
    private Double min;
    private Double max;
    private Integer n;
    private Double sumSqr;
    private Double sum;
    
    public BasicNumericInfoUnitState() {
        this.min= Double.POSITIVE_INFINITY;
        this.max= Double.NEGATIVE_INFINITY;
        this.n= 0;
        this.sum= 0.0;
        this.sumSqr= 0.0;
        
    }
    
    public UnitState getNewBlank() {
        return new BasicNumericInfoUnitState();
    }

    public void merge(UnitState state) throws Exception {
        if(!(state instanceof BasicNumericInfoUnitState))
            throw new Exception("Expected BasicNumericInfoUnitState, got " + state.getClass().toString());
        BasicNumericInfoUnitState newState= (BasicNumericInfoUnitState) state;
        this.max= Math.max(this.max, newState.getMax());
        this.min= Math.min(this.min, newState.getMin());
        this.n= this.n + newState.getN();
        this.sum= this.sum + newState.getSum();
        this.sumSqr= this.sumSqr + newState.getSumSqr();        
    }

    public void addData(Object value) {
        if(value instanceof Double) {
            addData((Double)value);
            return;
        }
        if(CommonUtils.isValidNumber(value)) {
            Double dVal= Double.valueOf(value.toString());
            addData(dVal);
        }
    }
    
    public void addData(Double dVal) {
        if(dVal==null)
            return;
        this.n++;
        this.sum+= dVal;
        this.sumSqr+= Math.pow(dVal, 2);
        this.max= Math.max(this.max, dVal);
        this.min= Math.min(this.min, dVal);
    }

    public Double getMin() {
        return this.min;
    }
    public Double getSumSqr() {
        return this.sumSqr;
    }
    public Double getSum() {
        return this.sum;
    }
    public Integer getN() {
        return this.n;
    }
    
    public Double getMax() {
        return this.max;
    }
    
    
    public void populateUnivariateStats(UnivariateStats univariateStats, Params params) {
        ContStats contStats= univariateStats.getContStats();
        if(contStats==null)
            contStats= new ContStats();
        
        contStats.setTotalValuesSum(this.sum);
        contStats.setTotalSquaresSum(this.sumSqr);
        univariateStats.withContStats(contStats);
        
        NumericInfo numInfo= univariateStats.getNumericInfo();
        if(numInfo==null)
            numInfo= new NumericInfo();
        numInfo.withMaximum(this.max);
        numInfo.withMinimum(this.min);
        if(n==0 || sum.isInfinite() || sumSqr.isInfinite()) {
            univariateStats.withNumericInfo(numInfo);
            return;
        }
        
        numInfo.setMean(sum/n);
        Double EPS= 1e-6;
        double stdDev = Math.sqrt((this.sumSqr - (this.sum * this.sum) / this.n + EPS)
                / (this.n - 1));
        numInfo.setStandardDeviation(stdDev);
        univariateStats.withNumericInfo(numInfo);
        // does not set median or quartile range
        
    }
    
}
