/*
 * Copyright [2013-2016] PayPal Software Foundation
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
import ml.shifu.shifu.core.dtrain.DTrainUtils;
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
     * (name, column config) map for quick index
     */
    private Map<String, ColumnConfig> columnConfigMap = new HashMap<String, ColumnConfig>();

    /**
     * Valid meta size which is in final output
     */
    private int validMetaSize = 0;

    public EvalNormUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        evalConfig = modelConfig.getEvalConfigByName(evalSetName);

        if(StringUtils.isBlank(evalConfig.getDataSet().getHeaderPath())) {
            throw new RuntimeException("The evaluation data set header couldn't be empty!");
        }

        this.headers = CommonUtils.getFinalHeaders(evalConfig);

        Set<String> evalNamesSet = new HashSet<String>(Arrays.asList(this.headers));
        this.outputNames = new ArrayList<String>();

        // 1. target at first
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getTargetColumnName())) {
            outputNames.add(evalConfig.getDataSet().getTargetColumnName());

        } else {
            outputNames.add(modelConfig.getTargetColumnName());
        }

        // 2. weight column
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())) {
            outputNames.add(evalConfig.getDataSet().getWeightColumnName());
        } else {
            outputNames.add("weight");
        }

        // 3. do populate columnConfigMap at first
        for(ColumnConfig columnConfig: this.columnConfigList) {
            columnConfigMap.put(columnConfig.getColumnName(), columnConfig);
        }

        // 4. add meta columns
        List<String> allMetaColumns = evalConfig.getAllMetaColumns(modelConfig);
        for(String meta: allMetaColumns) {
            if(evalNamesSet.contains(meta)) {
                outputNames.add(meta);
                validMetaSize += 1;
            } else {
                throw new RuntimeException("Meta variable - " + meta + " couldn't be found in eval dataset!");
            }
        }

        // 5. append real valid features
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        boolean isAfterVarSelect = (inputOutputIndex[3] == 1);
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(isAfterVarSelect) {
                if(columnConfig.isFinalSelect() && (!columnConfig.isMeta() && !columnConfig.isTarget())) {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        if(!outputNames.contains(columnConfig.getColumnName())) {
                            outputNames.add(columnConfig.getColumnName());
                        }
                    } else {
                        throw new RuntimeException("FinalSelect variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                    }
                }
            } else {
                if(!columnConfig.isMeta() && !columnConfig.isTarget()) {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        if(!outputNames.contains(columnConfig.getColumnName())) {
                            outputNames.add(columnConfig.getColumnName());
                        }
                    } else {
                        throw new RuntimeException("Variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                    }
                }
            }
        }
    }

    public Tuple exec(Tuple input) throws IOException {
        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(input, this.headers);

        Tuple tuple = TupleFactory.getInstance().newTuple(this.outputNames.size());
        for(int i = 0; i < this.outputNames.size(); i++) {
            String name = this.outputNames.get(i);
            String raw = rawDataMap.get(name);
            if(i == 0) {
                tuple.set(i, raw);
            } else if(i == 1) {
                tuple.set(i, (StringUtils.isEmpty(raw) ? "1" : raw));
            } else if(i > 1 && i < 2 + validMetaSize) {
                // [2, 2 + validMetaSize) are meta columns
                tuple.set(i, raw);
            } else {
                ColumnConfig columnConfig = this.columnConfigMap.get(name);
                Double value = Normalizer.normalize(columnConfig, raw, this.modelConfig.getNormalizeStdDevCutOff(),
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
            for(int i = 0; i < this.outputNames.size(); i++) {
                String name = this.outputNames.get(i);
                if(i < 2 + validMetaSize) {
                    // set target, weight and meta columns to string
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
