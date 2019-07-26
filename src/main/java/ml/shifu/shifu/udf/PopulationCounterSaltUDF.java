/*
 * Copyright [2012-2019] PayPal Software Foundation
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

import com.google.common.primitives.Longs;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.udf.stats.CategoryCounter;
import ml.shifu.shifu.udf.stats.Counter;
import ml.shifu.shifu.udf.stats.NumericCounter;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Calculate the counter for each bin
 */
public class PopulationCounterSaltUDF extends AbstractTrainerUDF<Tuple> {

    public static Logger logger = LoggerFactory.getLogger(PopulationCounterSaltUDF.class);

    private Counter counter;
    private int index;

    // DO NOT use this constructor
    private PopulationCounterSaltUDF(String source, String pathModelConfig, String pathColumnConfig)
        throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.index = 1;
    }

    public PopulationCounterSaltUDF(String source, String pathModelConfig, String pathColumnConfig, String index)
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
        logger.info("PopulationCounterSaltUDF: Start to count bin value count for {}, bag size {}", columnConfig.getColumnName(), bag.size());

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
                Boolean tag = (Boolean) tuple.get(index + 1);
                counter.addData(tag, (value == null) ? null : value.toString());
            }
        }

        Tuple output = TupleFactory.getInstance().newTuple(5);
        output.set(0, columnId);
        output.set(1, counter.getBinLen());
        output.set(2, counter.getUnitSum());
        output.set(3, StringUtils.join(Longs.asList(counter.getPositiveCounter()), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        output.set(4, StringUtils.join(Longs.asList(counter.getNegativeCounter()), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        return output;
    }


    public Schema outputSchema(Schema input) {
        try {
            return Utils
                .getSchemaFromString("PopulationInfo:Tuple(columnId : int, binLen : int, unitSum : double, positiveCounter : chararray, negativeCounter : chararray)");
        } catch (ParserException e) {
            log.error("Error when generating output schema.", e);
            // just ignore
            return null;
        }
    }
}
