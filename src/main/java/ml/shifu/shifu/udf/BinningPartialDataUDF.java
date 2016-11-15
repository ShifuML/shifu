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
import java.util.Iterator;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;
import ml.shifu.shifu.core.binning.EqualPopulationBinning;

import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.logicalLayer.schema.Schema;

/**
 * GenBinningDataUDF class
 * 
 * @Nov 11, 2014
 * 
 */
public class BinningPartialDataUDF extends AbstractTrainerUDF<String> {

    private int columnId = -1;
    private AbstractBinning<?> binning = null;
    private int histoScaleFactor;

    /**
     * @param source
     * @param pathModelConfig
     * @param pathColumnConfig
     * @throws IOException
     */
    public BinningPartialDataUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        this(source, pathModelConfig, pathColumnConfig, "100");
    }

    /**
     * @param source
     * @param pathModelConfig
     * @param pathColumnConfig
     * @throws IOException
     */
    public BinningPartialDataUDF(String source, String pathModelConfig, String pathColumnConfig, String histoScaleFactor)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.histoScaleFactor = NumberFormatUtils.getInt(histoScaleFactor, 100);
        if(this.histoScaleFactor < 100) {
            this.histoScaleFactor = 100;
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public String exec(Tuple input) throws IOException {
        if(input == null) {
            return null;
        }

        DataBag databag = (DataBag) input.get(0);
        Iterator<Tuple> iterator = databag.iterator();
        while(iterator.hasNext()) {
            Tuple element = iterator.next();
            if(element == null) {
                continue;
            }

            if(columnId < 0) {
                columnId = (Integer) element.get(0);
                ColumnConfig columnConfig = super.columnConfigList.get(columnId);
                if(columnConfig.isCategorical()) {
                    binning = new CategoricalBinning(-1);
                } else {
                    if(super.modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)) {
                        binning = new EqualIntervalBinning(modelConfig.getStats().getMaxNumBin());
                    } else {
                        binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin());
                    }
                }
            }

            Object value = element.get(1);
            if (value != null) {
                if (isWeightBinningMethod() && binning instanceof EqualPopulationBinning) {
                    ((EqualPopulationBinning) binning).addData(value.toString(), (Double) element.get(4));
                } else {
                    binning.addData(value.toString());
                }
            }
        }

        String binningObjStr = ((binning == null) ? null : binning.objToString());

        cleanUp();

        return binningObjStr;
    }

    private boolean isWeightBinningMethod(){
        return modelConfig.getBinningMethod().equals(BinningMethod.WeightTotal)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightInterval)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightPositive)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightNegative);
    }

    /**
     * cleanup the binning information
     */
    private void cleanUp() {
        this.columnId = -1;
        this.binning = null;
    }

    @Override
    public Schema outputSchema(Schema input) {
        return new Schema(new Schema.FieldSchema("binning_info", DataType.CHARARRAY));
    }
}
