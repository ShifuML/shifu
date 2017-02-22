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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;

import org.apache.pig.data.Tuple;

/**
 * FilterBinningDataUDF class
 * 
 * @author zhanhu
 */
public class FilterBinningDataUDF extends AbstractTrainerUDF<Boolean> {

    public FilterBinningDataUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Boolean exec(Tuple input) throws IOException {
        Integer columnNum = (Integer) input.get(0);
        if(columnNum == null) {
            return false;
        }
        ColumnConfig columnConfig = columnConfigList.get(columnNum);

        boolean isPositive = (Boolean) input.get(2);

        if(isValidRecord(modelConfig.isRegression(), isPositive, columnConfig)) {
            return true;
        }
        return false;
    }

    private boolean isValidRecord(boolean isBinary, boolean isPositive, ColumnConfig columnConfig) {
        if(isBinary) {
            return columnConfig != null
                    && (columnConfig.isCategorical()
                    || modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.DynamicBinning)
                    || modelConfig.getBinningMethod().equals(BinningMethod.EqualTotal)
                    || modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)
                    || (modelConfig.getBinningMethod().equals(BinningMethod.EqualPositive) && isPositive)
                    || (modelConfig.getBinningMethod().equals(BinningMethod.EqualNegtive) && !isPositive)
                    || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualTotal)
                    || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualInterval)
                    || (modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualPositive) && isPositive)
                    || (modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualNegative) && !isPositive));
        } else {
            return columnConfig != null
                    && ( columnConfig.isCategorical()
                    || modelConfig.getBinningMethod().equals(BinningMethod.EqualTotal)
                    || modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)
                    || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualTotal)
                    || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualInterval));
        }
    }
}
