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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.pmml.builder.impl.ZscoreLocalTransformCreator;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.encog.ml.BasicML;

/**
 * Calculate the score for each evaluation data
 */
public class EvalNormUDF extends AbstractTrainerUDF<Tuple> {

    @SuppressWarnings("unused")
    private static final String SCHEMA_PREFIX = "eval::";

    private EvalConfig evalConfig;
    private String[] headers;
    private List<String> outputNames;

    private String scoreName;
    private int scIndex;

    private ModelRunner modelRunner;
    private String scale;

    /**
     * (name, column config) map for quick index
     */
    private Map<String, ColumnConfig> columnConfigMap = new HashMap<String, ColumnConfig>();

    /**
     * Valid meta size which is in final output
     */
    private int validMetaSize = 0;

    public EvalNormUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName, String scale)
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

        // 3. add meta columns
        List<String> allMetaColumns = evalConfig.getAllMetaColumns(modelConfig);
        for(String meta: allMetaColumns) {
            if(evalNamesSet.contains(meta)) {
                if(!outputNames.contains(meta)) {
                    outputNames.add(meta);
                    validMetaSize += 1;
                }
            } else {
                throw new RuntimeException("Meta variable - " + meta + " couldn't be found in eval dataset!");
            }
        }

        // 4. do populate columnConfigMap at first
        for(ColumnConfig columnConfig: this.columnConfigList) {
            columnConfigMap.put(columnConfig.getColumnName(), columnConfig);
            if ( columnConfig.isFinalSelect() && !outputNames.contains(columnConfig.getColumnName())) {
                validMetaSize += 1;
                outputNames.add(columnConfig.getColumnName());
            }
        }

        // 5. append real valid features
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        boolean isAfterVarSelect = (inputOutputIndex[3] == 1);
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(isAfterVarSelect) {
                if(columnConfig.isFinalSelect() && (!columnConfig.isMeta() && !columnConfig.isTarget())) {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        outputNames.add(columnConfig.getColumnName());
                    } else {
                        throw new RuntimeException("FinalSelect variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                    }
                }
            } else {
                if(!columnConfig.isMeta() && !columnConfig.isTarget()) {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        //if(!outputNames.contains(columnConfig.getColumnName())) {
                            //outputNames.add(columnConfig.getColumnName());
                        outputNames.add(columnConfig.getColumnName());
                        //}
                    } else {
                        throw new RuntimeException("Variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                    }
                }
            }
        }

        this.scoreName = this.evalConfig.getPerformanceScoreSelector();
        if ( StringUtils.isBlank(this.scoreName) || this.scoreName.equalsIgnoreCase("mean")) {
            this.scIndex = -1;
        } else {
            try {
                this.scIndex = Integer.parseInt(this.scoreName.toLowerCase().replaceAll("model", ""));
            } catch (Exception e) {
                throw new RuntimeException("Invalid setting for performanceScoreSelector in EvalConfig - "
                        + this.scoreName);            }
        }
        this.scale = scale;
    }

    public Tuple exec(Tuple input) throws IOException {
        if(this.modelRunner == null) {
            // here to initialize modelRunner, this is moved from constructor to here to avoid OOM in client side.
            // UDF in pig client will be initialized to get some metadata issues
            List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, this.columnConfigList, evalConfig,
                    evalConfig.getDataSet().getSource(), evalConfig.getGbtConvertToProb());
            this.modelRunner = new ModelRunner(modelConfig, columnConfigList, this.headers, evalConfig.getDataSet()
                    .getDataDelimiter(), models);
            this.modelRunner.setScoreScale(Integer.parseInt(this.scale));
        }

        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(input, this.headers);

        Tuple tuple = TupleFactory.getInstance().newTuple(this.outputNames.size() + 1);
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

        CaseScoreResult score = this.modelRunner.compute(rawDataMap);
        if ( score == null ) {
            tuple.set(this.outputNames.size(), -999.0);
        } else if ( this.scIndex < 0 ) {
            tuple.set(this.outputNames.size(), score.getAvgScore());
        } else {
            tuple.set(this.outputNames.size(), score.getScores().get(this.scIndex));
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
                    tupleSchema.add(new FieldSchema(
                            ZscoreLocalTransformCreator.genPmmlColumnName(name, this.modelConfig.getNormalizeType()),
                            DataType.DOUBLE));
                }
            }
            tupleSchema.add(new FieldSchema(this.scoreName, DataType.DOUBLE));
            return new Schema(new FieldSchema("EvalNorm", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
