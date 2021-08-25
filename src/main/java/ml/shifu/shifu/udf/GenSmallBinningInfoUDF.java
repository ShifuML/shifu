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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;
import ml.shifu.shifu.core.binning.MunroPatBinning;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.Accumulator;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;

/**
 * Created by zhanhu on 7/5/16.
 */
public class GenSmallBinningInfoUDF extends AbstractTrainerUDF<Tuple> implements Accumulator<Tuple> {

    private static Logger logger = LoggerFactory.getLogger(GenSmallBinningInfoUDF.class);

    private int scaleFactor = 1024;

    private Integer columnId =  null;
    private ColumnConfig columnConfig = null;
    @SuppressWarnings("rawtypes")
    private AbstractBinning binning = null;

    public GenSmallBinningInfoUDF(String source, String pathModelConfig, String pathColumnConfig,
            String histoScaleFactor) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.scaleFactor = Integer.parseInt(histoScaleFactor);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() != 1) {
            return null;
        }

        accumulate(input);
        Tuple output = getValue();
        cleanup();

        return output;
    }

    private boolean isToBinningVal(ColumnConfig columnConfig, Boolean isPostive) {
        return columnConfig.isCategorical()
                || modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualTotal)
                || modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualInterval)
                || (modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualPositive) && isPostive)
                || (modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualNegative) && !isPostive)
                || modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.WeightEqualTotal)
                || modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.WeightEqualInterval)
                || (modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.WeightEqualPositive) && isPostive)
                || (modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.WeightEqualNegative) && !isPostive);
    }

    @SuppressWarnings("rawtypes")
    private AbstractBinning getBinningHandler(ColumnConfig columnConfig) {
        AbstractBinning binning = null;
        if(columnConfig.isNumerical()) {
            if(modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualInterval)) {
                binning = new EqualIntervalBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
            } else {
                binning = new MunroPatBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
            }
        } else {
            binning = new CategoricalBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues(),
                    this.maxCategorySize,columnConfig.getHashSeed());
        }

        return binning;
    }

    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new Schema.FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new Schema.FieldSchema("bins", DataType.CHARARRAY));

            return new Schema(new Schema.FieldSchema("binning", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

    @Override
    public void accumulate(Tuple input) throws IOException {
        logger.info("Start to accumulate for small binning ... ");

        DataBag dataBag = (DataBag) input.get(0);
        Iterator<Tuple> iterator = dataBag.iterator();
        while(iterator.hasNext()) {
            Tuple tuple = iterator.next();
            if(tuple != null && tuple.size() >= 3) {
                if(columnId == null) {
                    columnId = (Integer) tuple.get(0);
                    if(columnId >= super.columnConfigList.size()) {
                        int newColumnId = columnId % super.columnConfigList.size();
                        columnConfig = super.columnConfigList.get(newColumnId);
                    } else {
                        columnConfig = super.columnConfigList.get(columnId);
                    }

                    binning = getBinningHandler(columnConfig);
                }

                Boolean isPositive = (Boolean) tuple.get(2);
                if(isToBinningVal(columnConfig, isPositive)) {
                    String val = (String) tuple.get(1);
                    binning.addData(val);
                }
            }
        }
    }

    @Override
    public Tuple getValue() {
        Tuple output = TupleFactory.getInstance().newTuple();
        output.append(columnId);
        output.append(StringUtils.join(this.binning.getDataBin(), AbstractBinning.FIELD_SEPARATOR));
        return output;
    }

    @Override
    public void cleanup() {
        this.columnId = null;
        this.columnConfig = null;
        this.binning = null;
    }
}
