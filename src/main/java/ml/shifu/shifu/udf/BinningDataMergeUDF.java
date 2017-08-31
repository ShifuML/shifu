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
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;

/**
 * MergeBinningDataUDF class
 */
public class BinningDataMergeUDF extends AbstractTrainerUDF<Tuple> {

    public BinningDataMergeUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Tuple exec(Tuple input) throws IOException {
        if(input == null) {
            return null;
        }

        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);
        int corrColumnId = columnId;
        if(corrColumnId >= super.columnConfigList.size()){
            corrColumnId = corrColumnId % super.columnConfigList.size();
        }
        ColumnConfig columnConfig = super.columnConfigList.get(corrColumnId);

        AbstractBinning<?> binning = null;
        AbstractBinning<?> backupBinning = null;
        log.info("Start merging bin info for columnId - " + columnId + ", the bag size is - " + databag.size());

        Iterator<Tuple> iterator = databag.iterator();
        while(iterator.hasNext()) {
            Tuple element = iterator.next();
            if(element == null || element.size() < 2) {
                continue;
            }

            String objValStr = (String) element.get(1);
            String hybridCateValStr = null;

            long start = System.currentTimeMillis();

            // for hybrid, split
            if(columnConfig.isHybrid()) {
                String[] splits = CommonUtils.split(objValStr, Constants.HYBRID_BIN_STR_DILIMETER);
                objValStr = splits[0];
                hybridCateValStr = splits[1];
            }
            AbstractBinning<?> partialBinning = AbstractBinning.constructBinningFromStr(modelConfig, columnConfig,
                    objValStr);
            AbstractBinning<?> partialBackupBinning = null;
            if(columnConfig.isHybrid()) {
                partialBackupBinning = new CategoricalBinning();
                partialBackupBinning.stringToObj(hybridCateValStr);
            }
            log.info("constructBinningFromStr: " + (System.currentTimeMillis() - start) + "ms");
            start = System.currentTimeMillis();

            if(binning == null) {
                binning = partialBinning;
                if(columnConfig.isHybrid()) {
                    backupBinning = partialBackupBinning;
                }
            } else {
                binning.mergeBin(partialBinning);
                if(columnConfig.isHybrid()) {
                    backupBinning.mergeBin(partialBackupBinning);
                }
            }
            log.info("mergeBin: " + (System.currentTimeMillis() - start) + "ms");
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        List<?> binFields = binning.getDataBin();

        // Do check here. It's because if there are too many value for categorical variable,
        // it will consume too much memory when join them together, that will cause OOM exception
        if(columnConfig.isCategorical() && binFields.size() > this.maxCategorySize) {
            log.warn(columnId + " " + columnConfig.getColumnName() + " is over maximal categorical size: "
                    + this.maxCategorySize);
            output.set(1, "");
        } else {
            if(columnConfig.isHybrid()) {
                String finalBinStr = StringUtils.join(binFields, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
                finalBinStr += Constants.HYBRID_BIN_STR_DILIMETER
                        + StringUtils.join(backupBinning.getDataBin(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
                output.set(1, finalBinStr);
            } else {
                output.set(1, StringUtils.join(binFields, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
            }
        }

        log.info("Finish merging bin info for columnId - " + columnId);

        return output;
    }

    @Override
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("binningDataInfo", DataType.CHARARRAY));

            return new Schema(new Schema.FieldSchema("BinningDataInfo", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
