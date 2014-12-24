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

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;

/**
 * MergeBinningDataUDF class
 * 
 * @Nov 11, 2014
 *
 */
public class BinningDataMergeUDF extends AbstractTrainerUDF<Tuple> {

    /**
     * @param source
     * @param pathModelConfig
     * @param pathColumnConfig
     * @throws IOException
     */
    public BinningDataMergeUDF(String source, String pathModelConfig, String pathColumnConfig)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    /* (non-Javadoc)
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Tuple exec(Tuple input) throws IOException {
        if ( input == null ) {
            return null;
        }
        
        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);
        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        
        AbstractBinning<?> binning = null;
        log.info("Start merging bin info for columnId - " + columnId + ", the bag size is - " + databag.size());
        
        Iterator<Tuple> iterator = databag.iterator();
        while ( iterator.hasNext() ) {
            Tuple element = iterator.next();
            if ( element == null || element.size() < 2) {
                continue;
            }
            
            String objValStr = (String) element.get(1);
            AbstractBinning<?> partialBinning = AbstractBinning.constructBinningFromStr(modelConfig, columnConfig, objValStr);
            if ( binning == null ) {
                binning = partialBinning;
            } else {
                binning.mergeBin(partialBinning);
            }
        }
        
        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        List<?> binFields = binning.getDataBin();

        // Do check here. It's because if there are too many value for categorical variable,
        // it will consume too much memory when join them together, that will cause OOM exception
        if ( binFields.size() > CalculateNewStatsUDF.MAX_CATEGORICAL_BINC_COUNT ) {
            output.set(1, "");
        } else {
            output.set(1, StringUtils.join(binning.getDataBin(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
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
