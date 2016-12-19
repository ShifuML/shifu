/**
 * Copyright [2012-2014] PayPal Software Foundation
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
import java.util.Random;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.Constants;

import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.tools.pigstats.PigStatusReporter;

/**
 * <pre>
 * AddColumnNumUDF class is to convert tuple of row data into bag of column data
 * Its structure is like
 *    {
 *         (column-id, column-value, column-tag, column-score)
 *         (column-id, column-value, column-tag, column-score)
 *         ...
 * }
 */
public class AddColumnNumAndFilterUDF extends AbstractTrainerUDF<DataBag> {

    private static final int TOTAL_COLUMN_CNT   = 4;
    private static final int COLUMN_ID_INDX     = 0;
    private static final int COLUMN_VAL_INDX    = 1;
    private static final int COLUMN_TAG_INDX    = 2;
    private static final int COLUMN_SEED_INDX   = 3;


    private Random random = new Random(System.currentTimeMillis());

    private final boolean isAppendRandom;

    public AddColumnNumAndFilterUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr)
            throws Exception {
        this(source, pathModelConfig, pathColumnConfig, withScoreStr, "true");
    }

    public AddColumnNumAndFilterUDF(String source, String pathModelConfig, String pathColumnConfig,
            String withScoreStr, String isAppendRandom) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);
        this.isAppendRandom = Boolean.TRUE.toString().equalsIgnoreCase(isAppendRandom);
    }

    @SuppressWarnings("deprecation")
    public DataBag exec(Tuple input) throws IOException {
        DataBag bag = BagFactory.getInstance().newDefaultBag();
        TupleFactory tupleFactory = TupleFactory.getInstance();

        if(input == null) {
            return null;
        }

        int size = input.size();

        if(size == 0 || input.size() < this.columnConfigList.size()) {
            log.info("the input size - " + input.size() + ", while column size - " + columnConfigList.size());
            throw new ShifuException(ShifuErrorCode.ERROR_NO_EQUAL_COLCONFIG);
        }

        if(input.get(tagColumnNum) == null) {
            throw new ShifuException(ShifuErrorCode.ERROR_NO_TARGET_COLUMN);
        }

        String tag = input.get(tagColumnNum).toString();

        // filter out tag not in setting tagging list
        if(!super.tagSet.contains(tag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        Double rate = modelConfig.getBinningSampleRate();
        if(modelConfig.isBinningSampleNegOnly()) {
            if(super.negTagSet.contains(tag) && random.nextDouble() > rate) {
                return null;
            }
        } else {
            if(random.nextDouble() > rate) {
                return null;
            }
        }

        for(int i = 0; i < size; i++) {
            ColumnConfig config = columnConfigList.get(i);
            if(config.isCandidate()) {
                boolean isPositive = false;;
                if(modelConfig.isRegression()) {
                    if ( super.posTagSet.contains(tag) ) {
                        isPositive = true;
                    } else if(super.negTagSet.contains(tag)) {
                        isPositive = false;
                    } else {
                        // not valid tag, just skip current record
                        continue;
                    }
                }
                if(!isValidRecord(modelConfig.isRegression(), isPositive, config)) {
                    continue;
                }
                Tuple tuple = tupleFactory.newTuple(TOTAL_COLUMN_CNT);
                tuple.set(COLUMN_ID_INDX, i);
                // Set Data
                tuple.set(COLUMN_VAL_INDX, (input.get(i) == null ? null : input.get(i).toString()));

                if(modelConfig.isRegression()) {
                    // Set Tag
                    if(super.posTagSet.contains(tag)) {
                        tuple.set(COLUMN_TAG_INDX, true);
                    }

                    if(super.negTagSet.contains(tag)) {
                        tuple.set(COLUMN_TAG_INDX, false);
                    }
                } else {
                    // a mock for multiple classification
                    tuple.set(COLUMN_TAG_INDX, true);
                }

                // add random seed for distribution for bigger mapper, 300 is not enough TODO
                if(this.isAppendRandom) {
                    tuple.set(COLUMN_SEED_INDX, Math.abs(random.nextInt() % 300));
                }
                bag.add(tuple);
            }
        }
        return bag;
    }

    @Override
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("value", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("tag", DataType.BOOLEAN));
            if(this.isAppendRandom) {
                tupleSchema.add(new FieldSchema("rand", DataType.INTEGER));
            }
            return new Schema(new Schema.FieldSchema("columnInfos", new Schema(new Schema.FieldSchema("columnInfo",
                    tupleSchema, DataType.TUPLE)), DataType.BAG));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

    private boolean isValidRecord(boolean isBinary, boolean isPositive, ColumnConfig columnConfig) {
        if(isBinary) {
            return columnConfig != null
                    && ( columnConfig.isCategorical()
                        || modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.DynamicBinning)
                        || modelConfig.getBinningMethod().equals(BinningMethod.EqualTotal)
                        || modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)
                        || (modelConfig.getBinningMethod().equals(BinningMethod.EqualPositive) && isPositive)
                        || (modelConfig.getBinningMethod().equals(BinningMethod.EqualNegtive) && !isPositive));
        } else {
            return columnConfig != null
                    && ( columnConfig.isCategorical()
                        || modelConfig.getBinningMethod().equals(BinningMethod.EqualTotal)
                        || modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval));
        }
    }
}
