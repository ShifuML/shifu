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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.udf.CalculateStatsUDF;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CategoricalVarStats class
 * 
 * @Nov 3, 2014
 *
 */
public class CategoricalVarStats extends AbstractVarStats {
    
    private static Logger log = LoggerFactory.getLogger(CategoricalVarStats.class);
    private Map<String, Integer> categoricalBinMap;
    
    /**
     * @param modelConfig
     * @param columnConfig
     * @param valueThreshold
     */
    public CategoricalVarStats(ModelConfig modelConfig, ColumnConfig columnConfig, Double valueThreshold) {
        super(modelConfig, columnConfig, valueThreshold);
    }
    
    /* (non-Javadoc)
     * @see ml.shifu.shifu.udf.stats.AbstractVarStats#runVarStats(java.lang.String, org.apache.pig.data.DataBag)
     */
    @Override
    public void runVarStats(String binningInfo, DataBag databag) throws ExecException {
        String[] binningDataArr = StringUtils.split(binningInfo, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        
        log.info("Column Name - " + this.columnConfig.getColumnName() + ", Column Bin Length - " + binningDataArr.length);
        
        columnConfig.setBinCategory(Arrays.asList(binningDataArr));
        categoricalBinMap = new HashMap<String, Integer>(columnConfig.getBinCategory().size());
        for ( int i = 0; i < columnConfig.getBinCategory().size(); i ++ ) {
            categoricalBinMap.put(columnConfig.getBinCategory().get(i), Integer.valueOf(i));
        }
        
        statsCategoricalColumnInfo(databag, columnConfig);
    }
    
    /**
     * @param databag
     * @param columnConfig
     * @throws ExecException 
     */
    private void statsCategoricalColumnInfo(DataBag databag, ColumnConfig columnConfig) throws ExecException {
        Integer[] binCountPos = new Integer[columnConfig.getBinCategory().size()];
        Integer[] binCountNeg = new Integer[columnConfig.getBinCategory().size()];
        Double[] binWeightCountPos = new Double[columnConfig.getBinCategory().size()];
        Double[] binWeightCountNeg = new Double[columnConfig.getBinCategory().size()];
        
        initializeZeroArr(binCountPos);
        initializeZeroArr(binCountNeg);
        initializeZeroArr(binWeightCountPos);
        initializeZeroArr(binWeightCountNeg);
        
        Iterator<Tuple> iterator = databag.iterator();
        while ( iterator.hasNext() ) {
            Tuple element = iterator.next();
            
            if ( element.size() != 4 ) {
                continue;
            }
            
            Object value = element.get(1);
            String tag = (String) element.get(2);
            Double weight = (Double) element.get(3);
            
            if ( value == null || StringUtils.isBlank(value.toString()) ) {
                //TODO check missing value list in ModelConfig??
                missingValueCnt ++;
                continue;
            }
            String str = StringUtils.trim(value.toString());
            
            // int binNum = CommonUtils.getBinNum(columnConfig, str);
            int binNum = quickLocateCategorialBin(str);
            if ( binNum < 0 ) {
                continue;
            }
            
            if ( modelConfig.getPosTags().contains(tag) ) {
                increaseInstCnt(binCountPos, binNum);
                increaseInstCnt(binWeightCountPos, binNum, weight);
            } else if ( modelConfig.getNegTags().contains(tag) ) {
                increaseInstCnt(binCountNeg, binNum);
                increaseInstCnt(binWeightCountNeg, binNum, weight);
            }
        }
        
        columnConfig.setBinCountPos(Arrays.asList(binCountPos));
        columnConfig.setBinCountNeg(Arrays.asList(binCountNeg));
        columnConfig.setBinWeightedPos(Arrays.asList(binWeightCountPos));
        columnConfig.setBinWeightedNeg(Arrays.asList(binWeightCountNeg));
        
        calculateBinPosRateAndAvgScore();
        
        for ( int i = 0; i < columnConfig.getBinCountPos().size(); i ++ ) {
            int posCount = columnConfig.getBinCountPos().get(i);
            int negCount = columnConfig.getBinCountNeg().get(i);
            
            binning.addData(columnConfig.getBinPosRate().get(i), posCount);
            binning.addData(columnConfig.getBinPosRate().get(i), negCount);
            
            streamStatsCalculator.addData(columnConfig.getBinPosRate().get(i), posCount);
            streamStatsCalculator.addData(columnConfig.getBinPosRate().get(i), negCount);
        }
        
        columnConfig.setMax(streamStatsCalculator.getMax());
        columnConfig.setMean(streamStatsCalculator.getMean());
        columnConfig.setMin(streamStatsCalculator.getMin());
        if ( binning.getMedian() == null ) {
            columnConfig.setMedian(streamStatsCalculator.getMean());
        } else {
            columnConfig.setMedian(binning.getMedian());
        }
        columnConfig.setStdDev(streamStatsCalculator.getStdDev());
        
        // Currently, invalid value will be regarded as missing
        columnConfig.setMissingCnt(missingValueCnt + invalidValueCnt);
        columnConfig.setTotalCount(databag.size());
        columnConfig.setMissingPercentage(((double)columnConfig.getMissingCount()) / columnConfig.getTotalCount());
    }

    /**
     * @param val
     * @return
     */
    private int quickLocateCategorialBin(String val) {
        Integer binNum = categoricalBinMap.get(val);
        return ((binNum == null) ? -1 : binNum);
    }

}
