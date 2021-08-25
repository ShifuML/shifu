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

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.udf.stats.CategoryCounter;
import ml.shifu.shifu.udf.stats.Counter;
import ml.shifu.shifu.udf.stats.NumericCounter;

/**
 * Calculate the counter for each bin
 */
public class PopulationCounterSaltSumUDF extends AbstractTrainerUDF<Tuple> {

    public static Logger logger = LoggerFactory.getLogger(PopulationCounterSaltSumUDF.class);

    private Counter counter;
    @SuppressWarnings("unused")
    private int index;

    // DO NOT use this constructor
    private PopulationCounterSaltSumUDF(String source, String pathModelConfig, String pathColumnConfig)
        throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.index = 1;
    }

    public PopulationCounterSaltSumUDF(String source, String pathModelConfig, String pathColumnConfig, String index)
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
        DataBag dataBag = (DataBag) input.get(1);
        Integer columnId = (Integer) groupInfo.get(1);
        ColumnConfig columnConfig = columnConfigList.get(columnId);
        logger.info("PopulationCounterSaltSumUDF: Start to count bin value count for {}, bag size {}", columnConfig.getColumnName(), dataBag.size());

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

        Iterator<Tuple> iter = dataBag.iterator();
        while (iter.hasNext()) {
            Tuple tuple = iter.next();
            if (tuple != null && tuple.size() != 0 && tuple.size() > 3) {
                Tuple counters = (Tuple)tuple.get(3);
                if(counters.size() > 4) {
                    double unitNum = (Double) counters.get(2);
                    String positiveStr = (String) counters.get(3);
                    String negativeStr = (String) counters.get(4);
                    counter.setUnitSum(counter.getUnitSum() + unitNum);
                    counter.setPositiveCounter(mergeLongList(counter.getPositiveCounter(), positiveStr));
                    counter.setNegativeCounter(mergeLongList(counter.getNegativeCounter(), negativeStr));
                }
            }
        }

        List<Long> dataBin = counter.getCounter();
        Tuple output = TupleFactory.getInstance().newTuple(4);
        output.set(0, columnId);
        String unit = (groupInfo.get(0) == null ? "" : groupInfo.get(0).toString());
        output.set(1, StringUtils.join(dataBin, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        output.set(2,  toStatsText(unit, counter.getUnitMean(),
                counter.getMissingRate(), counter.getTotalInstCnt(), counter.getDistMetrics()));
        output.set(3, unit);
        return output;
    }

    private long[] mergeLongList(long[] totalPositive, String positiveStr){
        String [] positiveList = StringUtils.split(positiveStr, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        for(int i = 0; i < totalPositive.length; i++){
            totalPositive[i] += Long.parseLong(positiveList[i]);
        }
        return totalPositive;
    }

    public Schema outputSchema(Schema input) {
        try {
            return Utils
                .getSchemaFromString("PopulationInfo:Tuple(columnId : int, population : chararray, unitstats : chararray, psiColumn : chararray)");
        } catch (ParserException e) {
            log.error("Error when generating output schema.", e);
            // just ignore
            return null;
        }
    }

    private String toStatsText(String unit, double mean, double missingRate, long totalInstCnt, ColumnStatsCalculator.ColumnMetrics metrics) {
        return unit
                + "^" + Double.toString(mean)
                + "^" + Double.toString(missingRate)
                + "^" + Long.toString(totalInstCnt)
                + "^" + (metrics == null ? 0.0d : metrics.getIv())
                + "^" + (metrics == null ? 0.0d : metrics.getKs());

    }
}
