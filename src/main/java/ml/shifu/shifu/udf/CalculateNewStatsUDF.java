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
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.KSIVCalculator;
import ml.shifu.shifu.core.StreamStatsCalculator;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

/**
 * CalculateNewStatsUDF class
 * 
 * @author zhanhu
 * @Oct 27, 2014
 *
 */
public class CalculateNewStatsUDF extends AbstractTrainerUDF<Tuple> {

    private Double valueThreshold = 1e6;

    public CalculateNewStatsUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        if (modelConfig.getNumericalValueThreshold() != null) {
            valueThreshold = modelConfig.getNumericalValueThreshold();
        }
        log.debug("Value Threshold: " + valueThreshold);
    }

    /* (non-Javadoc)
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Tuple exec(Tuple input) throws IOException {
        if (input == null || input.size() != 2) {
            return null;
        }
        
        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);
        String binningDataInfo = (String) input.get(3);
        
        log.info("column 2 - " + input.get(2).toString());
        log.info("column id - " + columnId + ", " + binningDataInfo);
        
        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        
        String[] binningDataArr = StringUtils.split(binningDataInfo, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        if ( columnConfig.isCategorical() ) {
            columnConfig.setBinCategory(Arrays.asList(binningDataArr));
            statsCategoricalColumnInfo(databag, columnConfig);
        } else {
            List<Double> binBoundary = new ArrayList<Double>();
            for ( String binningInfo : binningDataArr ) {
                 binBoundary.add(Double.valueOf(binningInfo));
            }
            columnConfig.setBinBoundary(binBoundary);
            statsNumericalColumnInfo(databag, columnConfig);
        }
        
        
        KSIVCalculator ksivCalculator = new KSIVCalculator();
        ksivCalculator.calculateKSIV(columnConfig.getBinCountNeg(), columnConfig.getBinCountPos());
        
        // Assemble the results
        DecimalFormat df = new DecimalFormat("##.######");

        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append(columnId);
        if ( columnConfig.isCategorical() ) {
            tuple.append("[" + StringUtils.join(columnConfig.getBinCategory(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR) + "]");
        } else {
            tuple.append(columnConfig.getBinBoundary().toString());
        }
        
        tuple.append(columnConfig.getBinCountNeg().toString());
        tuple.append(columnConfig.getBinCountPos().toString());
        tuple.append(columnConfig.getBinAvgScore().toString());
        tuple.append(columnConfig.getBinPosRate().toString());
        
        tuple.append(df.format(ksivCalculator.getKS()));
        tuple.append(df.format(ksivCalculator.getIV()));

        tuple.append(df.format(columnConfig.getColumnStats().getMax()));
        tuple.append(df.format(columnConfig.getColumnStats().getMin()));
        tuple.append(df.format(columnConfig.getColumnStats().getMean()));
        tuple.append(df.format(columnConfig.getColumnStats().getStdDev()));

        if ( columnConfig.isCategorical() ) {
            tuple.append("C");
        } else {
            tuple.append("N");
        }

        tuple.append(columnConfig.getColumnStats().getMedian());
        tuple.append(df.format(0));
        tuple.append(df.format(databag.size()));
        tuple.append(df.format(0.0));
        tuple.append(columnConfig.getBinCountNeg().toString());
        tuple.append(columnConfig.getBinCountPos().toString());


        return tuple;
    }

    /**
     * @param databag
     * @param columnConfig
     * @throws ExecException 
     */
    private void statsCategoricalColumnInfo(DataBag databag, ColumnConfig columnConfig) throws ExecException {
        Integer[] binCountPos = new Integer[columnConfig.getBinCategory().size()];
        Integer[] binCountNeg = new Integer[columnConfig.getBinCategory().size()];
        initializeZeroArr(binCountPos);
        initializeZeroArr(binCountNeg);
        
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
            
            int binNum = CommonUtils.getBinNum(columnConfig, str);
            if ( modelConfig.getPosTags().contains(tag) ) {
                increaseInstCnt(binCountPos, binNum);
            } else if ( modelConfig.getNegTags().contains(tag) ) {
                increaseInstCnt(binCountNeg, binNum);
            }
        }
        
        columnConfig.setBinCountPos(Arrays.asList(binCountPos));
        columnConfig.setBinCountNeg(Arrays.asList(binCountNeg));
        
        calculateBinPosRateAndAvgScore(columnConfig);
        
        StreamStatsCalculator streamStatsCalculator = new StreamStatsCalculator(valueThreshold);
        for ( int i = 0; i < columnConfig.getBinCountPos().size(); i ++ ) {
            int posCount = columnConfig.getBinCountPos().get(i);
            int negCount = columnConfig.getBinCountNeg().get(i);
            
            for ( int j = 0; j < posCount ; j ++ ) {
                streamStatsCalculator.addData(columnConfig.getBinPosRate().get(i));
            }
            
            for ( int j = 0; j < negCount ; j ++ ) {
                streamStatsCalculator.addData(columnConfig.getBinPosRate().get(i));
            }
        }
        
        columnConfig.setMax(streamStatsCalculator.getMax());
        columnConfig.setMean(streamStatsCalculator.getMean());
        columnConfig.setMin(streamStatsCalculator.getMin());
        columnConfig.setMedian(Double.NaN);
        columnConfig.setStdDev(streamStatsCalculator.getStdDev());
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
                // invalid value
                log.debug("Invalid value - " + str, e);
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
        
        calculateBinPosRateAndAvgScore(columnConfig);
    }

    /**
     * @param arr
     */
    private void initializeZeroArr(Integer[] arr) {
        for ( int i = 0; i < arr.length; i ++ ) {
            arr[i] = Integer.valueOf(0);
        }
    }

    /**
     * @param binCountArr
     * @param binNum
     */
    private void increaseInstCnt(Integer[] binCountArr, int binNum) {
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
    private void calculateBinPosRateAndAvgScore(ColumnConfig columnConfig) {
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
}
