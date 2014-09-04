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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.ContStats;
import org.dmg.pmml.Interval;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.core.di.builtin.EqualPositiveColumnNumBinningCalculator;
import ml.shifu.core.di.builtin.KSIVCalculator;
import ml.shifu.core.di.builtin.QuantileCalculator;
import ml.shifu.core.di.builtin.WOECalculator;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.SerializedNumericalValueObject;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

/**
 * Expects data in the form of a SerializedNumericalValueObject
 */

public class BinomialRSampleUnitState extends RSampleUnitState<SerializedNumericalValueObject> implements UnitState {
    
    private static final long serialVersionUID = 1L;
    int numBins;
    
    public BinomialRSampleUnitState(int maximumSize, int numberBins) {
        super(maximumSize);
        numBins= numberBins;
    }
    
    public UnitState getNewBlank() {
        return new BinomialRSampleUnitState(maxSize, numBins);
    }

    public void merge(UnitState state) throws Exception {
        if(!(state instanceof BinomialRSampleUnitState))
            throw new Exception("Expected BinomialRSampleUnitState, got " + state.getClass().toString());
        super.merge((BinomialRSampleUnitState) state); 
        
    }

    public void addData(Object value) {
        super.addSample((SerializedNumericalValueObject) value);
    }
    
    public void populateUnivariateStats(UnivariateStats univariateStats,
            Params params) {

        ContStats contStats= univariateStats.getContStats();
        if(contStats==null)
            contStats= new ContStats();
        
        // TODO: use DI for NumBinningCalculator?
        EqualPositiveColumnNumBinningCalculator calculator = new EqualPositiveColumnNumBinningCalculator();
        // create List of NumericalValueObjects
        List<NumericalValueObject> nvoList= new ArrayList<NumericalValueObject>();
        for(SerializedNumericalValueObject sample: samples) {
            NumericalValueObject nvo= new NumericalValueObject();
            nvo.setIsPositive(sample.getIsPositive());
            nvo.setValue(sample.getValue());
            nvo.setWeight(sample.getWeight());
            nvoList.add(nvo);
        }
        
        ColumnBinningResult result = calculator.calculate(nvoList, numBins);

        int size = result.getLength();

        List<Interval> intervals = new ArrayList<Interval>();
        for (int i = 0; i < size; i++) {
            Interval interval = new Interval();
            interval.setClosure(Interval.Closure.OPEN_CLOSED);
            interval.setLeftMargin(result.getBinBoundary().get(i));
            if (i == size - 1) {
                interval.setRightMargin(Double.POSITIVE_INFINITY);
            } else {
                interval.setRightMargin(result.getBinBoundary().get(i + 1));
            }

            intervals.add(interval);

        }
        
        contStats.withIntervals(intervals);

        Map<String, String> extensionMap = new HashMap<String, String>();

        extensionMap.put("BinCountPos", result.getBinCountPos().toString());
        extensionMap.put("BinCountNeg", result.getBinCountNeg().toString());
        extensionMap.put("BinWeightedCountPos", result.getBinWeightedPos().toString());
        extensionMap.put("BinWeightedCountNeg", result.getBinWeightedNeg().toString());
        extensionMap.put("BinPosRate", result.getBinPosRate().toString());

        List<Double> woe = WOECalculator.calculate(result.getBinCountPos().toArray(), result.getBinCountNeg().toArray());
        extensionMap.put("BinWOE", woe.toString());

        KSIVCalculator ksivCalculator = new KSIVCalculator();
        ksivCalculator.calculateKSIV(result.getBinCountNeg(), result.getBinCountPos());
        extensionMap.put("KS", Double.valueOf(ksivCalculator.getKS()).toString());
        extensionMap.put("IV", Double.valueOf(ksivCalculator.getIV()).toString());

        contStats.withExtensions(PMMLUtils.createExtensions(extensionMap));
        univariateStats.withContStats(contStats);
        
        // Compute NumericInfo stats
        NumericInfo numInfo= univariateStats.getNumericInfo();
        if(numInfo==null)
            numInfo= new NumericInfo();
        
        List<Double> doubleSamples= new ArrayList<Double>();
        for(SerializedNumericalValueObject sample: samples)
            doubleSamples.add(sample.getValue());
        Collections.sort(doubleSamples);
        int numQuantiles= Integer.parseInt(params.get("numQuantiles", "11").toString());  
        QuantileCalculator quantileCalculator = new QuantileCalculator();
        numInfo.withQuantiles(quantileCalculator.getEvenlySpacedQuantiles(doubleSamples, numQuantiles));
        int sampleSize= doubleSamples.size();
        if(sampleSize!=0)
            numInfo.withInterQuartileRange(doubleSamples.get((int) Math.floor(sampleSize * 0.75)) - doubleSamples.get((int) Math.floor(sampleSize * 0.25)));
        numInfo.withMedian(doubleSamples.get(doubleSamples.size()/2));
        univariateStats.withNumericInfo(numInfo);

    }


}
