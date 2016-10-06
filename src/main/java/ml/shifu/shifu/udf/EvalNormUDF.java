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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;

import java.io.IOException;
import java.util.*;

/**
 * Calculate the score for each evaluation data
 */
public class EvalNormUDF extends AbstractTrainerUDF<Tuple> {

    private static final String SCHEMA_PREFIX = "eval::";

    private EvalConfig evalConfig;
    private String[] headers;
    private List<String> outputNames;

    /**
     * A simple weight exception validation: if over 5000 throw exceptions
     */
    private int weightExceptions;

    public EvalNormUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        evalConfig = modelConfig.getEvalConfigByName(evalSetName);

        // create model runner
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            this.headers = CommonUtils.getHeaders(
                    evalConfig.getDataSet().getHeaderPath(),
                    evalConfig.getDataSet().getHeaderDelimiter(),
                    evalConfig.getDataSet().getSource());

            Set<String> evalNamesSet = new HashSet<String>(Arrays.asList(this.headers));
            this.outputNames = new ArrayList<String>();

            if ( StringUtils.isNotBlank(evalConfig.getDataSet().getTargetColumnName()) ) {
                outputNames.add(evalConfig.getDataSet().getTargetColumnName());
            } else {
                outputNames.add(modelConfig.getWeightColumnName());
            }

            if ( StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName()) ) {
                outputNames.add(evalConfig.getDataSet().getWeightColumnName());
            } else {
                outputNames.add(SCHEMA_PREFIX + "weight");
            }

            for (ColumnConfig columnConfig : this.columnConfigList) {
                if ( columnConfig.isFinalSelect() ) {
                    if ( !evalNamesSet.contains(columnConfig.getColumnName()) ) {
                        log.error("FinalSelect variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                        throw new RuntimeException("FinalSelect variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                    } else {
                        outputNames.add(columnConfig.getColumnName());
                    }
                }
            }
        } else {
            log.error("The header couldn't be empty!");
            throw new RuntimeException("The evaluation data set header couldn't be empty!");
        }
    }

    public Tuple exec(Tuple input) throws IOException {
        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(input, this.headers);

        Tuple tuple = TupleFactory.getInstance().newTuple(this.outputNames.size());
        for ( int i = 0; i < this.outputNames.size(); i ++ ) {
            String name = this.outputNames.get(i);
            String raw = rawDataMap.get(name);
            if ( i == 0 ) {
                tuple.set(i, raw);
            } else if ( i == 1 ) {
                tuple.set(i, (StringUtils.isEmpty(raw) ? "1.0" : raw));
            } else {
                ColumnConfig columnConfig = CommonUtils.findColumnConfigByName(this.columnConfigList, name);
                Double value = Normalizer.normalize(columnConfig, raw,
                        this.modelConfig.getNormalizeStdDevCutOff(),
                        this.modelConfig.getNormalizeType());
                tuple.set(i, value);
            }
        }

        return tuple;
    }

    /**
     * output the schema for evaluation score
     */
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            for ( int i = 0; i < this.outputNames.size(); i ++ ) {
                String name = this.outputNames.get(i);
                if ( i < 2 ) {
                    tupleSchema.add(new FieldSchema(name, DataType.CHARARRAY));
                } else {
                    tupleSchema.add(new FieldSchema(name, DataType.DOUBLE));
                }
            }
            return new Schema(new FieldSchema("EvalNorm", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
