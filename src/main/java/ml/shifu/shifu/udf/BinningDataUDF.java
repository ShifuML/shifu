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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;
import ml.shifu.shifu.core.binning.EqualPopulationBinning;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.Accumulator;
import org.apache.pig.backend.executionengine.ExecException;
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
 * @Oct 27, 2014
 *
 */
public class BinningDataUDF extends AbstractTrainerUDF<Tuple> implements Accumulator<Tuple> {

    private int columnId = -1;
    private AbstractBinning<?> binning = null;
    
    /* (non-Javadoc)
     * @see org.apache.pig.Accumulator#accumulate(org.apache.pig.data.Tuple)
     */
    @Override
    public void accumulate(Tuple input) throws IOException {
        if ( input == null || input.size() != 2 ) {
            return;
        }
        
        columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);
        
        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        if ( binning == null ) {
            if ( columnConfig.isCategorical() ) {
                binning = new CategoricalBinning(-1);
            } else {
                if ( super.modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval) ) {
                    binning = new EqualIntervalBinning(modelConfig.getStats().getMaxNumBin());
                } else {
                    binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin());
                }
            }
        }
        
        Iterator<Tuple> iterator = databag.iterator();
        while ( iterator.hasNext() ) {
            Tuple element = iterator.next();
            if ( element == null || element.size() != 4) {
                continue;
            }
            
            Object value = element.get(1);
            if ( value != null ) {
                binning.addData(value.toString());
            }
        }
    }

    /* (non-Javadoc)
     * @see org.apache.pig.Accumulator#getValue()
     */
    @Override
    public Tuple getValue() {
        Tuple output = TupleFactory.getInstance().newTuple(2);
        try {
            output.set(0, columnId);
            output.set(1, StringUtils.join(binning.getDataBin(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        } catch ( ExecException e ) {
            log.error("Fail to generate output for columnId - " + columnId);
        }
        
        return output;
    }

    /* (non-Javadoc)
     * @see org.apache.pig.Accumulator#cleanup()
     */
    @Override
    public void cleanup() {
        this.columnId = -1;
        this.binning = null;
    }
    
    /**
     * @param source
     * @param pathModelConfig
     * @param pathColumnConfig
     * @throws IOException
     */
    public BinningDataUDF(String source, String pathModelConfig, String pathColumnConfig)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    /* (non-Javadoc)
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Tuple exec(Tuple input) throws IOException {
        if ( input == null || input.size() != 2 ) {
            return null;
        }
        
        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);
        
        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        AbstractBinning<?> binning = null;
        if ( columnConfig.isCategorical() ) {
            binning = new CategoricalBinning(-1);
        } else {
            if ( super.modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval) ) {
                binning = new EqualIntervalBinning(modelConfig.getStats().getMaxNumBin());
            } else {
                binning = new EqualPopulationBinning(modelConfig.getStats().getMaxNumBin());
            }
        }
        
        Iterator<Tuple> iterator = databag.iterator();
        while ( iterator.hasNext() ) {
            Tuple element = iterator.next();
            if ( element == null || element.size() != 4) {
                continue;
            }
            
            Object value = element.get(1);
            if ( value != null ) {
                binning.addData(value.toString());
            }
        }
        
        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        output.set(1, StringUtils.join(binning.getDataBin(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        
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
