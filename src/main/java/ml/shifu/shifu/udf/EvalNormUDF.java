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

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.impl.util.UDFContext;
import org.encog.ml.BasicML;

/**
 * Calculate the score for each evaluation data
 */
public class EvalNormUDF extends AbstractTrainerUDF<Tuple> {

    @SuppressWarnings("unused")
    private static final String SCHEMA_PREFIX = "eval::";
    private static final String ORIG_POSTFIX = "_orig";

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

    /**
     * If output raw variables together with norm variables
     */
    private boolean isOutputRaw = false;

    /**
     * Splits for filter expressions
     */
    private int segFilterSize = 0;

    /**
     * If append model score at last column
     */
    private boolean isAppendScore = false;

    public EvalNormUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName, String scale)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        evalConfig = modelConfig.getEvalConfigByName(evalSetName);

        if(!evalConfig.getNormAllColumns()) {
            // log such un compactiable
            log.warn("Default behanior is changed in eval norm to only norm selected columns.");
        }

        if(StringUtils.isBlank(evalConfig.getDataSet().getHeaderPath())) {
            log.warn("eval header path is empty, take the first line as schema (for csv format)");
        }

        this.headers = CommonUtils.getFinalHeaders(evalConfig);

        String filterExpressions = "";
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            filterExpressions = UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_SEGMENT_EXPRESSIONS);
        } else {
            filterExpressions = Environment.getProperty(Constants.SHIFU_SEGMENT_EXPRESSIONS);
        }
        if(StringUtils.isNotBlank(filterExpressions)) {
            this.segFilterSize = CommonUtils.split(filterExpressions,
                    Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER).length;
        }

        String isAppendScoreStr = "false";
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            isAppendScoreStr = UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_EVAL_NORM_APPEND_SCORE);
        } else {
            isAppendScoreStr = Environment.getProperty(Constants.SHIFU_EVAL_NORM_APPEND_SCORE);
        }

        if(StringUtils.isNotBlank(isAppendScoreStr)) {
            isAppendScore = isAppendScoreStr.equalsIgnoreCase(Boolean.TRUE.toString());
        } else {
            isAppendScore = false;
        }

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
        }

        // 5. append real valid features
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        boolean isAfterVarSelect = (inputOutputIndex[3] == 1);
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(isAfterVarSelect) {
                if(columnConfig.isFinalSelect() && (!columnConfig.isMeta() && !columnConfig.isTarget())) {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        // if has variables finalSelect=true and normAllColumns is false
                        if(!evalConfig.getNormAllColumns() && columnConfig.isFinalSelect()) {
                            if(!outputNames.contains(columnConfig.getColumnName())) {
                                outputNames.add(columnConfig.getColumnName());
                            }
                        }
                    } else {
                        throw new RuntimeException("FinalSelect variable - " + columnConfig.getColumnName()
                                + " couldn't be found in eval dataset!");
                    }
                }
            } else {
                if(!columnConfig.isMeta() && !columnConfig.isTarget()) {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        // no variable selected, no matter evalConfig.isNormAllColumns() true or false, just select all
                        // columns
                        if(!outputNames.contains(columnConfig.getColumnName())) {
                            outputNames.add(columnConfig.getColumnName());
                        }
                    } else {
                        throw new RuntimeException(
                                "Variable - " + columnConfig.getColumnName() + " couldn't be found in eval dataset!");
                    }
                }
            }
        }

        this.scoreName = this.evalConfig.getPerformanceScoreSelector();
        if(StringUtils.isBlank(this.scoreName) || this.scoreName.equalsIgnoreCase("mean")) {
            this.scIndex = -1;
        } else {
            try {
                this.scIndex = Integer.parseInt(this.scoreName.toLowerCase().replaceAll("model", ""));
            } catch (Exception e) {
                throw new RuntimeException(
                        "Invalid setting for performanceScoreSelector in EvalConfig - " + this.scoreName);
            }
        }
        this.scale = scale;

        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.isOutputRaw = Boolean.TRUE.toString().equalsIgnoreCase(UDFContext.getUDFContext().getJobConf()
                    .get(Constants.SHIFU_EVAL_NORM_OUTPUTRAW, Boolean.FALSE.toString()));
        } else {
            this.isOutputRaw = Boolean.TRUE.toString().equalsIgnoreCase(
                    Environment.getProperty(Constants.SHIFU_EVAL_NORM_OUTPUTRAW, Boolean.FALSE.toString()));
        }
    }

    public Tuple exec(Tuple input) throws IOException {
        if(this.modelRunner == null && this.isAppendScore) {
            // here to initialize modelRunner, this is moved from constructor to here to avoid OOM in client side.
            // UDF in pig client will be initialized to get some metadata issues
            @SuppressWarnings("deprecation")
            List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, evalConfig,
                    evalConfig.getDataSet().getSource(), evalConfig.getGbtConvertToProb(),
                    evalConfig.getGbtScoreConvertStrategy());
            this.modelRunner = new ModelRunner(modelConfig, columnConfigList, this.headers,
                    evalConfig.getDataSet().getDataDelimiter(), models);
            this.modelRunner.setScoreScale(Integer.parseInt(this.scale));
        }

        Map<NSColumn, String> rawDataNsMap = CommonUtils.convertDataIntoNsMap(input, this.headers, this.segFilterSize);
        if(MapUtils.isEmpty(rawDataNsMap)) {
            return null;
        }

        Tuple tuple = TupleFactory.getInstance().newTuple();
        for(int i = 0; i < this.outputNames.size(); i++) {
            String name = this.outputNames.get(i);
            String raw = rawDataNsMap.get(new NSColumn(name));
            if(i == 0) {
                tuple.append(raw);
            } else if(i == 1) {
                tuple.append(StringUtils.isEmpty(raw) ? "1" : raw);
            } else if(i > 1 && i < 2 + validMetaSize) {
                // [2, 2 + validMetaSize) are meta columns
                tuple.append(raw);
            } else {
                ColumnConfig columnConfig = this.columnConfigMap.get(name);
                List<Double> normVals = Normalizer.normalize(columnConfig, raw,
                        this.modelConfig.getNormalizeStdDevCutOff(), this.modelConfig.getNormalizeType());
                if(this.isOutputRaw) {
                    tuple.append(raw);
                }
                for(Double normVal: normVals) {
                    tuple.append(normVal);
                }
            }
        }

        if(this.isAppendScore && this.modelRunner != null) {
            CaseScoreResult score = this.modelRunner.computeNsData(rawDataNsMap);
            if(this.modelRunner == null || this.modelRunner.getModelsCnt() == 0 || score == null) {
                tuple.append(-999.0);
            } else if(this.scIndex < 0) {
                tuple.append(score.getAvgScore());
            } else {
                tuple.append(score.getScores().get(this.scIndex));
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
                name = normColumnName(name);
                if(i < 2 + validMetaSize) {
                    // set target, weight and meta columns to string
                    tupleSchema.add(new FieldSchema(name, DataType.CHARARRAY));
                } else {
                    if(this.isOutputRaw) {
                        ColumnConfig columnConfig = this.columnConfigMap.get(name);
                        if(columnConfig.isNumerical()) {
                            tupleSchema.add(new FieldSchema(name + ORIG_POSTFIX, DataType.DOUBLE));
                        } else {
                            tupleSchema.add(new FieldSchema(name + ORIG_POSTFIX, DataType.CHARARRAY));
                        }
                    }
                    tupleSchema.add(new FieldSchema(name, DataType.DOUBLE));
                }
            }
            if(this.isAppendScore) {
                tupleSchema.add(new FieldSchema(StringUtils.isBlank(this.scoreName) ? "default_score" : this.scoreName,
                        DataType.DOUBLE));
            }
            return new Schema(new FieldSchema("EvalNorm", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
    
    /**
     * Some column name has illegal chars which are all be normed in shifu. Such as ' ', '/' ..., are changed to '_'.
     * 
     * @param columnName
     *            the column name to be normed
     * @return normed column name
     */
    public static String normColumnName(String columnName) {
        if(StringUtils.isBlank(columnName)) {
            return columnName;
        }
        // replace empty and / to _ to avoid pig column schema parsing issue, all columns with empty
        // char or / in its name in shifu will be replaced;
        String newColumnName = columnName.replaceAll(" ", "_");
        newColumnName = newColumnName.replaceAll("/", "_");
        return newColumnName;
    }
}
