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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.KSIVCalculator;
import ml.shifu.shifu.udf.stats.AbstractVarStats;
import ml.shifu.shifu.util.Base64Utils;

import org.apache.commons.lang.StringUtils;
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
        if (input == null ) {
            return null;
        }
        
        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);
        String binningDataInfo = (String) input.get(3);
        
        log.info("start to process column id - " + columnId.toString());
        
        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        AbstractVarStats varstats = AbstractVarStats.getVarStatsInst(modelConfig, columnConfig, valueThreshold);
        varstats.runVarStats(binningDataInfo, databag);
        
        log.info("after to process column id - " + columnId.toString());
        
        KSIVCalculator ksivCalculator = new KSIVCalculator();
        ksivCalculator.calculateKSIV(columnConfig.getBinCountNeg(), columnConfig.getBinCountPos());
        
        // Assemble the results
        DecimalFormat df = new DecimalFormat("##.######");

        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append(columnId);
        if ( columnConfig.isCategorical() ) {
            if ( columnConfig.getBinCategory().size() == 0 ) {
                return null;
            }
            
            String binCategory = "[" + StringUtils.join(columnConfig.getBinCategory(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR) + "]";
            tuple.append(Base64Utils.base64Encode(binCategory));
        } else {
            if ( columnConfig.getBinBoundary().size() == 1 ) {
                return null;
            }
            
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
        tuple.append(df.format(columnConfig.getColumnStats().getMedian()));
        
        tuple.append(columnConfig.getMissingCount());
        tuple.append(columnConfig.getTotalCount());
        tuple.append(df.format(columnConfig.getMissingPercentage()));
        
        tuple.append(columnConfig.getBinWeightedNeg().toString());
        tuple.append(columnConfig.getBinWeightedPos().toString());


        return tuple;
    }

}
