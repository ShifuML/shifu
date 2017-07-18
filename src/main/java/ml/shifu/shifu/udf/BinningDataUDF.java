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
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;
import ml.shifu.shifu.core.binning.EqualPopulationBinning;
import ml.shifu.shifu.core.binning.MunroPatBinning;
import ml.shifu.shifu.core.binning.NativeBinning;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;

/**
 * BinningDataUDF class
 * 
 * @author zhanhu
 */
public class BinningDataUDF extends AbstractTrainerUDF<Tuple> {

    public BinningDataUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() < 2) {
            return null;
        }

        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);

        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        AbstractBinning<?> binning = null;
        if(columnConfig.isCategorical()) {
            binning = new CategoricalBinning(-1, this.maxCategorySize);
        } else {
            if(super.modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)) {
                binning = new EqualIntervalBinning(modelConfig.getStats().getMaxNumBin());
            } else {
                switch(this.modelConfig.getBinningAlgorithm()) {
                    case Native:
                        log.info("Invoke Native binning method, memory cosuming!!");
                        // always merge bins
                        binning = new NativeBinning(modelConfig.getStats().getMaxNumBin(), true);
                        break;
                    case SPDT:
                    case SPDTI:
                        log.info("Invoke SPDT(Streaming Parallel Decision Tree) binning method, ");
                        binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin());
                        break;
                    case MunroPat:
                    case MunroPatI:
                        log.info("Invoke Munro & Paterson selecting algorithm");
                        binning = new MunroPatBinning(modelConfig.getStats().getMaxNumBin());
                        break;
                    default:
                        log.info("Default: Invoke Munro & Paterson selecting algorithm");
                        binning = new MunroPatBinning(modelConfig.getStats().getMaxNumBin());
                        break;
                }
            }
        }

        Iterator<Tuple> iterator = databag.iterator();
        while(iterator.hasNext()) {
            Tuple element = iterator.next();
            if(element == null || element.size() < 2) {
                continue;
            }

            Object value = element.get(1);
            if(value != null) {
                binning.addData(value.toString());
            }
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        // Do check here. It's because if there are too many value for categorical variable,
        // it will consume too much memory when join them together, that will cause OOM exception
        List<?> dataBin = binning.getDataBin();
        if(dataBin.size() > this.maxCategorySize) {
            output.set(1, "");
        } else {
            output.set(1, StringUtils.join(dataBin, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        }

        log.info("Finish merging bin info for columnId - " + columnId);

        return output;
    }

    public Schema outputSchema(Schema input) {
        try {
            return Utils.getSchemaFromString("BinningDataInfo:Tuple(columnId : int, binningDataInfo : chararray)");
        } catch (ParserException e) {
            log.debug("Error when generating output schema.", e);
            // just ignore
            return null;
        }
    }
}