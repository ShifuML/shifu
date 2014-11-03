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
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.StreamStatsCalculator;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.CommonUtils;

/**
 * NumericalVarStats class
 * 
 * @author zhanhu
 * @Nov 3, 2014
 *
 */
public class NumericalVarStats extends AbstractVarStats {

    /**
     * @param modelConfig
     * @param columnConfig
     * @param valueThreshold
     */
    public NumericalVarStats(ModelConfig modelConfig, ColumnConfig columnConfig, Double valueThreshold) {
        super(modelConfig, columnConfig, valueThreshold);
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.udf.stats.AbstractVarStats#runVarStats(java.lang.String, org.apache.pig.data.DataBag)
     */
    @Override
    public void runVarStats(String binningInfo, DataBag databag) throws ExecException {
        String[] binningDataArr = StringUtils.split(binningInfo, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);

        List<Double> binBoundary = new ArrayList<Double>();
        for(String binningElement: binningDataArr) {
            binBoundary.add(Double.valueOf(binningElement));
        }
        
        columnConfig.setBinBoundary(binBoundary);
        statsNumericalColumnInfo(databag, columnConfig);
    }
    
    /**
     * @param databag
     * @param columnConfig
     * @throws ExecException 
     */
    private void statsNumericalColumnInfo(DataBag databag, ColumnConfig columnConfig) throws ExecException {     
        Integer[] binCountPos = new Integer[columnConfig.getBinBoundary().size()];
        Integer[] binCountNeg = new Integer[columnConfig.getBinBoundary().size()];
        initializeZeroArr(binCountPos);
        initializeZeroArr(binCountNeg);
        
        StreamStatsCalculator streamStatsCalculator = new StreamStatsCalculator(valueThreshold);
        
        Iterator<Tuple> iterator = databag.iterator();
        while ( iterator.hasNext() ) {
            Tuple element = iterator.next();
            
            if ( element.size() != 4 ) {
                continue;
            }
            
            Object value = element.get(1);
            String tag = (String) element.get(2);
            
            if ( value == null || StringUtils.isBlank(value.toString()) ) {
                continue;
            }
            String str = StringUtils.trim(value.toString());
            
            double colVal = 0.0;
            try {
                colVal = Double.parseDouble(str);
            } catch ( Exception e ) {
                continue;
            }
            
            streamStatsCalculator.addData(colVal);
            
            int binNum = CommonUtils.getBinNum(columnConfig, str);
            
            if ( modelConfig.getPosTags().contains(tag) ) {
                increaseInstCnt(binCountPos, binNum);
            } else if ( modelConfig.getNegTags().contains(tag) ) {
                increaseInstCnt(binCountNeg, binNum);
            }
        }
        
        columnConfig.setBinCountPos(Arrays.asList(binCountPos));
        columnConfig.setBinCountNeg(Arrays.asList(binCountNeg));
        
        columnConfig.setMax(streamStatsCalculator.getMax());
        columnConfig.setMean(streamStatsCalculator.getMean());
        columnConfig.setMin(streamStatsCalculator.getMin());
        columnConfig.setMedian(Double.NaN);
        columnConfig.setStdDev(streamStatsCalculator.getStdDev());
        
        calculateBinPosRateAndAvgScore();
    }

}
