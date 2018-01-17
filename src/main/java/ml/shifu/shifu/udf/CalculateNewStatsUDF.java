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
import java.text.DecimalFormat;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;
import ml.shifu.shifu.udf.stats.AbstractVarStats;
import ml.shifu.shifu.util.Base64Utils;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;

/**
 * CalculateNewStatsUDF class
 */
public class CalculateNewStatsUDF extends AbstractTrainerUDF<Tuple> {

    private Double valueThreshold = 1e6;

    private DecimalFormat df = new DecimalFormat("##.######");

    public CalculateNewStatsUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        if(modelConfig.getNumericalValueThreshold() != null) {
            valueThreshold = modelConfig.getNumericalValueThreshold();
        }
        log.debug("Value Threshold: " + valueThreshold);
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
        String binningDataInfo = (String) input.get(3);

        log.info("start to process column id - " + columnId.toString());

        ColumnConfig columnConfig = super.columnConfigList.get(columnId);
        AbstractVarStats varstats = AbstractVarStats.getVarStatsInst(modelConfig, columnConfig, valueThreshold);
        varstats.runVarStats(binningDataInfo, databag);

        log.info("after to process column id - " + columnId.toString());

        ColumnMetrics columnCountMetrics = ColumnStatsCalculator.calculateColumnMetrics(
                columnConfig.getBinCountNeg(), columnConfig.getBinCountPos());

        ColumnMetrics columnWeightMetrics = ColumnStatsCalculator.calculateColumnMetrics(
                columnConfig.getBinWeightedNeg(), columnConfig.getBinWeightedPos());

        // Assemble the results
        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append(columnId);
        if(columnConfig.isCategorical()) {
            if(columnConfig.getBinCategory().size() == 0
                    || columnConfig.getBinCategory().size() > this.maxCategorySize) {
                return null;
            }

            String binCategory = "["
                    + StringUtils.join(columnConfig.getBinCategory(), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR) + "]";
            tuple.append(Base64Utils.base64Encode(binCategory));
        } else {
            if(columnConfig.getBinBoundary().size() == 1) {
                return null;
            }

            tuple.append(columnConfig.getBinBoundary().toString());
        }

        tuple.append(columnConfig.getBinCountNeg().toString());
        tuple.append(columnConfig.getBinCountPos().toString());
        tuple.append(columnConfig.getBinAvgScore().toString());
        tuple.append(columnConfig.getBinPosRate().toString());

        tuple.append(df.format(columnCountMetrics.getKs()));
        tuple.append(df.format(columnCountMetrics.getIv()));

        tuple.append(df.format(columnConfig.getColumnStats().getMax()));
        tuple.append(df.format(columnConfig.getColumnStats().getMin()));
        tuple.append(df.format(columnConfig.getColumnStats().getMean()));
        tuple.append(df.format(columnConfig.getColumnStats().getStdDev()));
        if(columnConfig.isCategorical()) {
            tuple.append("C");
        } else {
            tuple.append("N");
        }
        tuple.append(df.format(columnConfig.getColumnStats().getMedian()));

        tuple.append(columnConfig.getMissingCount());
        tuple.append(columnConfig.getTotalCount());
        tuple.append(df.format(columnConfig.getMissingPercentage()));

        tuple.append(columnConfig.getBinWeightedNeg().toString());
        tuple.append(columnConfig.getBinWeightedPos().toString());

        tuple.append(columnCountMetrics.getWoe());
        tuple.append(columnWeightMetrics.getWoe());

        tuple.append(df.format(columnWeightMetrics.getKs()));
        tuple.append(df.format(columnWeightMetrics.getIv()));

        tuple.append(columnCountMetrics.getBinningWoe().toString());
        tuple.append(columnWeightMetrics.getBinningWoe().toString());
        tuple.append(columnConfig.getColumnStats().getSkewness());
        tuple.append(columnConfig.getColumnStats().getKurtosis());

        return tuple;
    }

    @Override
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("binBoundary", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binCountNeg", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binCountPos", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binAvgScore", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binPosRate", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("ks", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("iv", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("max", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("min", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("mean", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("stddev", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("isCate", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("median", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("missingCount", DataType.LONG));
            tupleSchema.add(new FieldSchema("totalCount", DataType.LONG));
            tupleSchema.add(new FieldSchema("missingRatio", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binWeightedNeg", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binWeightedPos", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("woe", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("weightedWoe", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("weightedKs", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("weightedIv", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binWoe", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("binWeightedWoe", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("skewness", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("kurtosis", DataType.CHARARRAY));
            return new Schema(new Schema.FieldSchema("ColumnStatistics", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
