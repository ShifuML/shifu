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
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.logicalLayer.schema.Schema;

/**
 * GenBinningDataUDF class
 * 
 */
public class BinningPartialDataUDF extends AbstractTrainerUDF<String> {

    private int columnId = -1;
    private AbstractBinning<?> binning = null;

    // backup binning for hybrid columns
    private AbstractBinning<?> backUpbinning = null;
    private int histoScaleFactor;
    private ColumnConfig columnConfig;

    public BinningPartialDataUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        this(source, pathModelConfig, pathColumnConfig, "100");
    }

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
                if(columnId >= super.columnConfigList.size()){
                    columnId = columnId % super.columnConfigList.size();
                }
                columnConfig = super.columnConfigList.get(columnId);
                if(columnConfig.isHybrid()) {
                    if(super.modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)) {
                        binning = new EqualIntervalBinning(modelConfig.getStats().getMaxNumBin() > 0 ? modelConfig
                                .getStats().getMaxNumBin() : 1024);
                    } else {
                        binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin() > 0 ? modelConfig
                                .getStats().getMaxNumBin() : 1024);
                    }

                    this.backUpbinning = new CategoricalBinning(-1, this.maxCategorySize);
                } else if(columnConfig.isCategorical()) {
                    binning = new CategoricalBinning(-1, this.maxCategorySize);
                } else {
                    if(super.modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)) {
                        binning = new EqualIntervalBinning(modelConfig.getStats().getMaxNumBin() > 0 ? modelConfig
                                .getStats().getMaxNumBin() : 1024);
                    } else {
                        binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin() > 0 ? modelConfig
                                .getStats().getMaxNumBin() : 1024);
                    }
                }
            }

            Object value = element.get(1);
            if(value != null) {
                String valStr = value.toString();
                if(isWeightBinningMethod() && binning instanceof EqualPopulationBinning) {
                    ((EqualPopulationBinning) binning).addData(valStr,
                            (Double) element.get(AddColumnNumUDF.COLUMN_WEIGHT_INDX));
                } else {
                    binning.addData(valStr);
                }
                if(this.columnConfig.isHybrid()) {
                    // missing value and not number value go to categorical binning
                    double douVal = CommonUtils.parseNumber(valStr);
                    Double hybridThreshould = this.columnConfig.getHybridThreshold();
                    if(hybridThreshould == null) {
                        hybridThreshould = Double.NEGATIVE_INFINITY;
                    }
                    // douVal < hybridThreshould which will also be set to category
                    boolean isCategory = Double.isNaN(douVal) || douVal < hybridThreshould;
                    if(douVal < hybridThreshould) {
                        log.warn("douVal " + douVal + ", threshold " + hybridThreshould + ", column {}"
                                + columnConfig.getColumnName());
                    }
                    if(binning.isMissingVal(valStr) || isCategory) {
                        this.backUpbinning.addData(valStr);
                    }
                }
            }
        }
        String binningObjStr = ((binning == null) ? null : binning.objToString());

        if(this.columnConfig.isHybrid()) {
            binningObjStr += Constants.HYBRID_BIN_STR_DILIMETER + this.backUpbinning.objToString();
        }

        cleanUp();

        return binningObjStr;
    }

    private boolean isWeightBinningMethod() {
        return modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualTotal)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualInterval)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualPositive)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualNegative);
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
