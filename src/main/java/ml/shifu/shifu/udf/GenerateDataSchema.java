package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.ColumnConfig;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.io.IOException;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

/**
 * Generate the schema from List of ColumnConfig.
 * There is no reason to assume all data source is PigStorage
 */
public class GenerateDataSchema extends AbstractTrainerUDF<Tuple> {

    public GenerateDataSchema(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        return input;
    }

    /**
     * output the schema for evaluation score
     */
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            for (ColumnConfig columnConfig : this.columnConfigList) {
                tupleSchema.add(new Schema.FieldSchema(columnConfig.getColumnName(), DataType.CHARARRAY));
            }
            return new Schema(new Schema.FieldSchema(null, tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
}
