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

import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;

/**
 * AbstractVarStats class
 * 
 * @author zhanhu
 * @Nov 3, 2014
 *
 */
public abstract class AbstractVarStats {
    
    protected ModelConfig modelConfig;
    protected ColumnConfig columnConfig;
    protected Double valueThreshold;
    
    public AbstractVarStats(ModelConfig modelConfig, ColumnConfig columnConfig, Double valueThreshold) {
        this.modelConfig = modelConfig;
        this.columnConfig = columnConfig;
        this.valueThreshold = valueThreshold;
    }
    
    /**
     * @param arr
     */
    protected void initializeZeroArr(Integer[] arr) {
        for ( int i = 0; i < arr.length; i ++ ) {
            arr[i] = Integer.valueOf(0);
        }
    }
    
    /**
     * @param arr
     */
    protected void initializeZeroArr(Double[] arr) {
        for ( int i = 0; i < arr.length; i ++ ) {
            arr[i] = Double.valueOf(0.0);
        }
    }
    
    /**
     * @param binCountArr
     * @param binNum
     */
    protected void increaseInstCnt(Integer[] binCountArr, int binNum) {
        Integer cnt = binCountArr[binNum];
        if ( cnt == null ) {
            cnt = Integer.valueOf(1);
        } else {
            cnt = Integer.valueOf(cnt.intValue() + 1);
        }
        
        binCountArr[binNum] = cnt;
    }
    
    /**
     * @param columnConfig
     */
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
