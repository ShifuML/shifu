/**
 * Copyright [2012-2014] eBay Software Foundation
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
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * <pre>
 * AddColumnNumUDF class is to convert tuple of row data into bag of column data
 * Its structure is like
 *    {
 * 		(column-id, column-value, column-tag, column-score)
 * 		(column-id, column-value, column-tag, column-score)
 * 		...
 * }
 */
public class AddColumnNumUDF extends AbstractTrainerUDF<DataBag> {

    private List<String> negTags;

    private Random random = new Random(System.currentTimeMillis());

    private int weightedColumnNum = -1;

    public AddColumnNumUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr)
            throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        if(!StringUtils.isEmpty(this.modelConfig.getDataSet().getWeightColumnName())) {
            String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();

            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                if(config.getColumnName().equals(weightColumnName)) {
                    this.weightedColumnNum = i;
                    break;
                }
            }
        }

        negTags = modelConfig.getNegTags();
    }

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
            if(modelConfig.isCategoricalDisabled()) {
                try {
                    Double.valueOf(input.get(i).toString());
                } catch (Exception e) {
                    continue;
                }
            }

            ColumnConfig config = columnConfigList.get(i);
            if(config.isCandidate()) {
                Tuple tuple = tupleFactory.newTuple(5);
                tuple.set(0, i);

                // Set Data
                tuple.set(1, input.get(i));

                // Set Tag
                tuple.set(2, tag);

                // set weights
                if(weightedColumnNum != -1) {
                    try {
                        tuple.set(3, Double.valueOf(input.get(weightedColumnNum).toString()));
                    } catch (NumberFormatException e) {
                        tuple.set(3, 1.0);
                    }

                    if(i == weightedColumnNum) {
                        // weight and its column, set to 1
                        tuple.set(3, 1.0);
                    }
                } else {
                    tuple.set(3, 1.0);
                }

                // add random seed for distribution
                tuple.set(4, Math.abs(random.nextInt() % 300));

                bag.add(tuple);
            }
        }

        return bag;
    }

    public Schema outputSchema(Schema input) {
        return null;
    }
}
