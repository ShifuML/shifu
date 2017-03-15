/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.udf;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.udf.stats.CategoryCounter;
import ml.shifu.shifu.udf.stats.Counter;
import ml.shifu.shifu.udf.stats.NumericCounter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Calculate the counter for each bin
 */
public class PopulationCounterUDF extends AbstractTrainerUDF<Tuple> {

    public static Logger logger = LoggerFactory.getLogger(PopulationCounterUDF.class);

    private Counter counter;
    private int index;

    // DO NOT use this constructor
    private PopulationCounterUDF(String source, String pathModelConfig, String pathColumnConfig)
        throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.index = 1;
    }

    public PopulationCounterUDF(String source, String pathModelConfig, String pathColumnConfig, String index)
        throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.index = Integer.valueOf(index);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return null;
        }

        Tuple groupInfo = (Tuple) input.get(0);
        DataBag bag = (DataBag) input.get(1);

        Integer columnId = (Integer) groupInfo.get(1);
        ColumnConfig columnConfig = columnConfigList.get(columnId);

        logger.info("Start to count bin value count for {}", columnConfig.getColumnName());

        if ( columnConfig.isCategorical()
                && CollectionUtils.isNotEmpty(columnConfig.getBinCategory()) ) {
            this.counter = new CategoryCounter(modelConfig.getMissingOrInvalidValues(),
                    columnConfig.getBinCategory(), columnConfig.getBinPosRate());
        } else if (columnConfig.isNumerical()
                && CollectionUtils.isNotEmpty(columnConfig.getBinBoundary()) ) {
            this.counter = new NumericCounter(modelConfig.getMissingOrInvalidValues(),
                    columnConfig.getColumnName(), columnConfig.getBinBoundary());
        } else {
            return null;
        }

        Iterator<Tuple> iter = bag.iterator();
        while (iter.hasNext()) {
            Tuple tuple = iter.next();
            if (tuple != null && tuple.size() != 0) {
                Object value = tuple.get(index);
                counter.addData((value == null) ? null : value.toString());
            }
        }

        List<Long> dataBin = counter.getCounter();

        Tuple output = TupleFactory.getInstance().newTuple(3);
        output.set(0, columnId);
        output.set(1, StringUtils.join(dataBin, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));

        String unit = (groupInfo.get(0) == null ? "" : groupInfo.get(0).toString());
        output.set(2,  toStatsText(unit, counter.getUnitMean(), counter.getMissingRate(), counter.getTotalInstCnt()));

        return output;
    }

    public Schema outputSchema(Schema input) {
        try {
            return Utils
                .getSchemaFromString("PopulationInfo:Tuple(columnId : int, population : chararray, unitstats : chararray)");
        } catch (ParserException e) {
            log.debug("Error when generating output schema.", e);
            // just ignore
            return null;
        }
    }

    private String toStatsText(String unit, double mean, double missingRate, long totalInstCnt) {
        return unit
                + "^" + Double.toString(mean)
                + "^" + Double.toString(missingRate)
                + "^" + Long.toString(totalInstCnt);
    }
}
