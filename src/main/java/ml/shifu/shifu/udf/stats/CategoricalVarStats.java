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

import java.util.*;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.udf.CalculateStatsUDF;

import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CategoricalVarStats class
 */
public class CategoricalVarStats extends AbstractVarStats {

    private static Logger log = LoggerFactory.getLogger(CategoricalVarStats.class);
    private Map<String, Integer> categoricalBinMap;

    public CategoricalVarStats(ModelConfig modelConfig, ColumnConfig columnConfig, Double valueThreshold) {
        super(modelConfig, columnConfig, valueThreshold);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.udf.stats.AbstractVarStats#runVarStats(java.lang.String, org.apache.pig.data.DataBag)
     */
    @Override
    public void runVarStats(String binningInfo, DataBag databag) throws ExecException {
        String[] binningDataArr = StringUtils.split(binningInfo, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);

        log.info("Column Name - " + this.columnConfig.getColumnName() + ", Column Bin Length - "
                + binningDataArr.length);

        columnConfig.setBinCategory(Arrays.asList(binningDataArr));
        categoricalBinMap = new HashMap<String, Integer>(columnConfig.getBinCategory().size());
        for(int i = 0; i < columnConfig.getBinCategory().size(); i++) {
            List<String> catValues = CommonUtils.flattenCatValGrp(columnConfig.getBinCategory().get(i));
            for ( String cval : catValues ) {
                categoricalBinMap.put(cval, Integer.valueOf(i));
            }
        }

        statsCategoricalColumnInfo(databag, columnConfig);
    }

    /**
     * @param databag
     * @param columnConfig
     * @throws ExecException
     */
    private void statsCategoricalColumnInfo(DataBag databag, ColumnConfig columnConfig) throws ExecException {
        // The last bin is for missingOrInvalid values
        Integer[] binCountPos = new Integer[columnConfig.getBinCategory().size() + 1];
        Integer[] binCountNeg = new Integer[columnConfig.getBinCategory().size() + 1];
        Double[] binWeightCountPos = new Double[columnConfig.getBinCategory().size() + 1];
        Double[] binWeightCountNeg = new Double[columnConfig.getBinCategory().size() + 1];
        int lastBinIndex = columnConfig.getBinCategory().size();
        initializeZeroArr(binCountPos);
        initializeZeroArr(binCountNeg);
        initializeZeroArr(binWeightCountPos);
        initializeZeroArr(binWeightCountNeg);

        Iterator<Tuple> iterator = databag.iterator();
        boolean isMissingValue = false;
        boolean isInvalidValue = false;
        while(iterator.hasNext()) {
            isInvalidValue = false;
            isMissingValue = false;
            Tuple element = iterator.next();

            if(element.size() < 4) {
                continue;
            }

            Object value = element.get(1);
            String tag = CommonUtils.trimTag((String) element.get(2));
            Double weight = (Double) element.get(3);

            int binNum = 0;

            if(value == null
                    || modelConfig.getDataSet().getMissingOrInvalidValues()
                            .contains(value.toString().toLowerCase().trim())) {
                // TODO check missing value list in ModelConfig??
                missingValueCnt++;
                isMissingValue = true;
            } else {
                String str = StringUtils.trim(value.toString());
                binNum = quickLocateCategorialBin(str);
                if(binNum < 0) {
                    invalidValueCnt++;
                    isInvalidValue = true;
                }
            }

            if(isInvalidValue || isMissingValue) {
                binNum = lastBinIndex;
            }

            if(modelConfig.getPosTags().contains(tag)) {
                increaseInstCnt(binCountPos, binNum);
                increaseInstCnt(binWeightCountPos, binNum, weight);
            } else if(modelConfig.getNegTags().contains(tag)) {
                increaseInstCnt(binCountNeg, binNum);
                increaseInstCnt(binWeightCountNeg, binNum, weight);
            }
        }

        columnConfig.setBinCountPos(Arrays.asList(binCountPos));
        columnConfig.setBinCountNeg(Arrays.asList(binCountNeg));
        columnConfig.setBinWeightedPos(Arrays.asList(binWeightCountPos));
        columnConfig.setBinWeightedNeg(Arrays.asList(binWeightCountNeg));

        calculateBinPosRateAndAvgScore();

        for(int i = 0; i < columnConfig.getBinCountPos().size(); i++) {
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
        if(binning.getMedian() == null) {
            columnConfig.setMedian(streamStatsCalculator.getMean());
        } else {
            columnConfig.setMedian(binning.getMedian());
        }
        columnConfig.setStdDev(streamStatsCalculator.getStdDev());

        // Currently, invalid value will be regarded as missing
        columnConfig.setMissingCnt(missingValueCnt + invalidValueCnt);
        columnConfig.setTotalCount(databag.size());
        columnConfig.setMissingPercentage(((double) columnConfig.getMissingCount()) / columnConfig.getTotalCount());
        columnConfig.getColumnStats().setSkewness(streamStatsCalculator.getSkewness());
        columnConfig.getColumnStats().setKurtosis(streamStatsCalculator.getKurtosis());
    }

    private int quickLocateCategorialBin(String val) {
        Integer binNum = categoricalBinMap.get(val);
        return ((binNum == null) ? -1 : binNum);
    }

    public static void main(String[] args) {
        System.out.println(Math.log((9 * 1.0d / 21d) / (41 * 1.0d / 90d)));
    }

}
