/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.udf.stats;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.StreamStatsCalculator;
import ml.shifu.shifu.core.binning.EqualPopulationBinning;

import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;

/**
 * AbstractVarStats class
 */
public abstract class AbstractVarStats {
    
    protected ModelConfig modelConfig;
    protected ColumnConfig columnConfig;
    protected Double valueThreshold;
    protected StreamStatsCalculator streamStatsCalculator;
    protected EqualPopulationBinning binning;
    
    protected long missingValueCnt = 0;
    protected long invalidValueCnt = 0;
    protected long totalValueCnt = 0;
    
    public AbstractVarStats(ModelConfig modelConfig, ColumnConfig columnConfig, Double valueThreshold) {
        this.modelConfig = modelConfig;
        this.columnConfig = columnConfig;
        this.valueThreshold = valueThreshold;
        
        this.streamStatsCalculator = new StreamStatsCalculator(valueThreshold);
        this.binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin());
    }
    
    protected void initializeZeroArr(Integer[] arr) {
        for ( int i = 0; i < arr.length; i ++ ) {
            arr[i] = Integer.valueOf(0);
        }
    }
    
    protected void initializeZeroArr(Double[] arr) {
        for ( int i = 0; i < arr.length; i ++ ) {
            arr[i] = Double.valueOf(0.0);
        }
    }
    
    protected void increaseInstCnt(Integer[] binCountArr, int binNum) {
        Integer cnt = binCountArr[binNum];
        if ( cnt == null ) {
            cnt = Integer.valueOf(1);
        } else {
            cnt = Integer.valueOf(cnt.intValue() + 1);
        }
        
        binCountArr[binNum] = cnt;
    }
    
    protected void increaseInstCnt(Double[] binWeightCountArr, int binNum, double weight) {
        Double weightCount = binWeightCountArr[binNum];
        if ( weightCount == null ) {
            weightCount = Double.valueOf(weight);
        } else {
            weightCount = Double.valueOf(weightCount.doubleValue() + weight);
        }
        
        binWeightCountArr[binNum] = weightCount;
    }
    
    
    protected void calculateBinPosRateAndAvgScore() {
        List<Double> binPositiveRate = new ArrayList<Double>();
        
        for (int i = 0; i < columnConfig.getBinCountPos().size(); i++) {
            int binPosCount = columnConfig.getBinCountPos().get(i);
            int binNegCount = columnConfig.getBinCountNeg().get(i);
            
            if ( binPosCount + binNegCount == 0 ) {
                binPositiveRate.add(0.0);
            } else {
                binPositiveRate.add( ((double)binPosCount) / (binPosCount + binNegCount) );
            }
        }
        
        columnConfig.setBinPosCaseRate(binPositiveRate);
        
        List<Integer> binAvgScore = new ArrayList<Integer>();
        for ( int i = 0; i < columnConfig.getBinCountPos().size(); i++ ) {
            binAvgScore.add(0);
        }
        columnConfig.setBinAvgScore(binAvgScore);
    }
    
    public abstract void runVarStats(String binningInfo, DataBag databag) throws ExecException;
    
    public static AbstractVarStats getVarStatsInst(ModelConfig modelConfig, ColumnConfig columnConfig, Double valueThreshold) {
        if ( columnConfig == null ) {
            return null;
        }
        
        if ( columnConfig.isCategorical() ) {
            return new CategoricalVarStats(modelConfig, columnConfig, valueThreshold);
        } else {
            return new NumericalVarStats(modelConfig, columnConfig, valueThreshold);
        }
    }
}
