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
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.*;
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
public class AddColumnNumUDF extends AbstractTrainerUDF<DataBag> {

    protected Set<String> negTags;

    private Random random = new Random(System.currentTimeMillis());

    public AddColumnNumUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr)
            throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        negTags = new HashSet<String>(modelConfig.getNegTags());
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
            if(negTags.contains(tag) && random.nextDouble() > rate) {
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
                Tuple tuple = tupleFactory.newTuple(5);
                tuple.set(0, i);

                // Set Data
                tuple.set(1, input.get(i) == null ? null : input.get(i).toString());

                if(modelConfig.isRegression()) {
                    // Set Tag
                    if(super.posTagSet.contains(tag)) {
                        tuple.set(2, true);
                    }

                    if(super.negTagSet.contains(tag)) {
                        tuple.set(2, false);
                    }
                } else {
                    // a mock for multiple classification
                    tuple.set(2, true);
                }

                // add random seed for distribution
                tuple.set(3, Math.abs(random.nextInt() % 300));

                // get weight value
                tuple.set(4, getWeightValue(input));
                bag.add(tuple);
            }
        }

        return bag;
    }

    private double getWeightValue(Tuple input){
        double weight = 1.0;
        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())) {
            for(ColumnConfig columnConfig: columnConfigList) {
                if(columnConfig.getColumnName().equalsIgnoreCase(modelConfig.getWeightColumnName().trim())) {
                    int columnId = columnConfig.getColumnNum();
                    try {
                        weight = Double.parseDouble(((DataByteArray) input.get(columnId)).toString());
                    } catch (ExecException ignore) {
                    }
                    break;
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

            return new Schema(new Schema.FieldSchema("columnInfos", new Schema(new Schema.FieldSchema("columnInfo",
                    tupleSchema, DataType.TUPLE)), DataType.BAG));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
}
