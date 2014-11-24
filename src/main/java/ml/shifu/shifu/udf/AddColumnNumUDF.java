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

import java.io.IOException;

import ml.shifu.shifu.container.obj.ColumnConfig;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.Accumulator;
import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;

/**
 * <pre>
 * AddColumnNumUDF class is to convert tuple of row data into bag of column data
 * Its structure is like
 *    {
 * 		(column-id, column-value, column-tag, column-score)
 * 		(column-id, column-value, column-tag, column-score)
 * 		...
 *  }
 */
public class AddColumnNumUDF extends AbstractTrainerUDF<DataBag>{

    private int weightedColumnNum = -1;
    private DataBag bag;
    private static final TupleFactory tupleFactory = TupleFactory.getInstance();
    
    public AddColumnNumUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        if (!StringUtils.isEmpty(this.modelConfig.getDataSet().getWeightColumnName())) {
            String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();

            for (int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                if (config.getColumnName().equals(weightColumnName)) {
                    this.weightedColumnNum = i;
                    break;
                }
            }
        }

    }

    @Override
    public DataBag exec(Tuple input) throws IOException {

        if (bag != null) {
            bag.clear();
        }
        bag = BagFactory.getInstance().newDefaultBag();

        if (input == null) return null;

        int columnSize = input.size();

        if(columnSize == 0 || columnSize != columnConfigList.size()) {
            throw new RuntimeException("the input records length is not equal to the column config file");
        }

        if(input.get(tagColumnNum) == null) {
            log.warn("the target column is null");
            return null;
        }

        String tag = input.get(tagColumnNum).toString();

        for(int i = 0; i < columnSize; i ++){

            if (modelConfig.isCategoricalDisabled()) {
                try {
                    Double.valueOf(input.get(i).toString());
                } catch (Exception e) {
                    continue;
                }
            }

            ColumnConfig config = columnConfigList.get(i);

            if (config.isCandidate()) {

                Tuple tuple = tupleFactory.newTuple(4);

                tuple.set(0, i);
                tuple.set(1, input.get(i));
                tuple.set(2, tag);

                if (weightedColumnNum != -1) {
                    try {
                        tuple.set(3, Double.valueOf(input.get(weightedColumnNum).toString()));
                    } catch (NumberFormatException e) {
                        tuple.set(3, 1.0);
                    }

                    if (i == weightedColumnNum) {
                        //weight and its column, set to 1
                        tuple.set(3, 1.0);
                    }
                } else {
                    tuple.set(3, 1.0);
                }

                bag.add(tuple);
            }
        }

        return bag;
    }

    @Override
    public Schema outputSchema(Schema input) {
    	/*try {
            Schema tupleSchema = new Schema();
            

            tupleSchema.add(new FieldSchema("col", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("value", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("target", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("weight", DataType.DOUBLE));
            
            Schema bagSchema = new Schema(new Schema.FieldSchema("shifu_col", tupleSchema, DataType.TUPLE));
            
            return new Schema(new Schema.FieldSchema(null, bagSchema, DataType.BAG));

        } catch (Exception e) {
            e.printStackTrace();
        */    return null;
        //}
    }

}
