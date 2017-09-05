/*
 * Copyright [2012-2014] PayPal Software Foundation
 * <p/>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.*;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.tools.pigstats.PigStatusReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * <pre>
 * AddColumnNumUDF class is to convert tuple of row data into bag of column data
 * Its structure is like
 *    {
 *         (column-id, column-value, column-tag, column-score)
 *         (column-id, column-value, column-tag, column-score)
 *         ...
 * }
 * </pre>
 */
public class AddColumnNumUDF extends AbstractTrainerUDF<DataBag> {

    private static final Logger LOG = LoggerFactory.getLogger(AddColumnNumUDF.class);
    private static final int INVALID_INDEX = -1;

    public static final int TOTAL_COLUMN_CNT = 5;
    public static final int COLUMN_ID_INDX = 0;
    public static final int COLUMN_VAL_INDX = 1;
    public static final int COLUMN_TAG_INDX = 2;
    public static final int COLUMN_SEED_INDX = 3;
    public static final int COLUMN_WEIGHT_INDX = 4;

    protected Set<String> negTags;
    protected int weightColumnId = INVALID_INDEX;
    protected Random random = new Random(System.currentTimeMillis());

    public AddColumnNumUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr)
            throws Exception {
        super(source, pathModelConfig, pathColumnConfig);
        negTags = new HashSet<String>(modelConfig.getNegTags());
        // get weight column ID
        if(StringUtils.isNotBlank(this.modelConfig.getWeightColumnName())) {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.getColumnName().equalsIgnoreCase(modelConfig.getWeightColumnName().trim())) {
                    this.weightColumnId = columnConfig.getColumnNum();
                }
            }
        }
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
            log.info("tagColumnNum is " + tagColumnNum + "; input size is " + input.size()
                    + "; columnConfigList.size() is " + columnConfigList.size() + "; tuple is"
                    + input.toDelimitedString("|") + "; tag is " + input.get(tagColumnNum));
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        String tag = CommonUtils.trimTag(input.get(tagColumnNum).toString());

        // filter out tag not in setting tagging list
        if(!super.tagSet.contains(tag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        Double rate = modelConfig.getBinningSampleRate();
        if(modelConfig.isBinningSampleNegOnly()) {
            if(negTags.contains(tag) && random.nextDouble() > rate) {
                return null;
            }
        } else {
            if(random.nextDouble() > rate) {
                return null;
            }
        }

        for(int i = 0; i < size; i++) {
            Tuple tuple = tupleFactory.newTuple(TOTAL_COLUMN_CNT);
            tuple.set(COLUMN_ID_INDX, i);

            // Set Data
            tuple.set(COLUMN_VAL_INDX, input.get(i) == null ? null : input.get(i).toString());

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

            // add random seed for distribution
            tuple.set(COLUMN_SEED_INDX, Math.abs(random.nextInt() % 300));

            // get weight value
            tuple.set(COLUMN_WEIGHT_INDX, getWeightColumnVal(input));
            bag.add(tuple);
        }

        return bag;
    }

    @SuppressWarnings("deprecation")
    protected double getWeightColumnVal(Tuple input) {
        double weight = 1.0;
        if(this.weightColumnId != INVALID_INDEX) {
            try {
                weight = Double.parseDouble(input.get(this.weightColumnId).toString());
                if(weight < 0d) {
                    LOG.warn("weight column is less than 0.");
                    if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_WEIGHT_RECORDS")) {
                        PigStatusReporter.getInstance()
                                .getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_WEIGHT_RECORDS").increment(1);
                    }
                }
            } catch (Exception e) {
                LOG.warn("weight column is not numerical or null.");
                if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_WEIGHT_RECORDS")) {
                    PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_WEIGHT_RECORDS")
                            .increment(1);
                }
            }
        }
        return weight;
    }

    @Override
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("value", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("tag", DataType.BOOLEAN));
            tupleSchema.add(new FieldSchema("rand", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("weight", DataType.DOUBLE));
            return new Schema(new Schema.FieldSchema("columnInfos", new Schema(new Schema.FieldSchema("columnInfo",
                    tupleSchema, DataType.TUPLE)), DataType.BAG));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
}
