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

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.udf.norm.CategoryMissingNormType;
import ml.shifu.shifu.udf.norm.PrecisionType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.ModelSpecLoaderUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.impl.util.UDFContext;
import org.encog.ml.BasicML;

import java.io.IOException;
import java.util.*;

/**
 * Calculate the score for each evaluation data
 */
public class EvalNormUDF extends AbstractEvalUDF<Tuple> {

    private static final String ORIG_POSTFIX = "_orig";

    private String[] headers;
    // feature names maybe different from outputNames,
    // since one feature may generate multi output after normalization
    private List<String> featureNames;
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
     * For categorical feature, a map is used to save query time in execution
     */
    private Map<Integer, Map<String, Integer>> categoricalIndexMap = new HashMap<Integer, Map<String, Integer>>();

    /**
     * In Zscore norm type, how to process category default missing value norm, by default use mean, another option is
     * POSRATE.
     */
    private CategoryMissingNormType categoryMissingNormType = CategoryMissingNormType.POSRATE;

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

    /**
     * There is header for input or not?
     */
    private boolean isCsvFormat = false;

    private PrecisionType precisionType;

    public EvalNormUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName, String scale)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig, evalSetName);

        if(!evalConfig.getNormAllColumns()) {
            // log such incompatible
            log.warn("Default behavior is changed in eval norm to only norm selected columns.");
        }

        if(StringUtils.isBlank(evalConfig.getDataSet().getHeaderPath())) {
            log.warn("eval header path is empty, take the first line as schema (for csv format)");
            this.isCsvFormat = true;
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
        this.outputNames = new ArrayList<>();

        // 1. target at first
        outputNames.add(modelConfig.getTargetColumnName(evalConfig));

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

        // 4. build categorical index map
        for(ColumnConfig config: columnConfigList) {
            if(config.isCategorical()) {
                Map<String, Integer> map = new HashMap<String, Integer>();
                if(config.getBinCategory() != null) {
                    for(int i = 0; i < config.getBinCategory().size(); i++) {
                        List<String> catValues = CommonUtils.flattenCatValGrp(config.getBinCategory().get(i));
                        for(String cval: catValues) {
                            map.put(cval, i);
                        }
                    }
                }
                this.categoricalIndexMap.put(config.getColumnNum(), map);
            }
        }

        // 5. do populate columnConfigMap at first
        for(ColumnConfig columnConfig: this.columnConfigList) {
            columnConfigMap.put(columnConfig.getColumnName(), columnConfig);
        }

        this.featureNames = new ArrayList<>(this.outputNames); // will be different from here
        // 6. append real valid features
        boolean hasSelectedVars = DTrainUtils.hasFinalSelectedVars(this.columnConfigList);
        Set<Integer> modelFeatureSet = DTrainUtils.getModelFeatureSet(this.columnConfigList, hasSelectedVars,
                hasCandidates);
        appendModelFeatures(this.columnConfigList, modelFeatureSet, evalNamesSet, featureNames, outputNames,
                ((hasSelectedVars) ? "FinalSelect variable" : "Variable"), this.segFilterSize);

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

        setPrecisionType();
        setCategoryMissingNormType();
    }

    private void appendModelFeatures(List<ColumnConfig> columnConfigList, Set<Integer> modelFeatureSet,
            Set<String> evalNamesSet, List<String> featureNames, List<String> outputNames, String varDesc,
            int segFilterSize) {
        boolean isSeg = segFilterSize > 0; // if appeneding segment expression variables
        int rawCCSize = columnConfigList.size() / (segFilterSize + 1);
        for(ColumnConfig columnConfig: columnConfigList) {
            if(modelFeatureSet.contains(columnConfig.getColumnNum())) {
                if(isSeg) {
                    if(columnConfig.getColumnNum() < rawCCSize) { // raw varaibles
                        if(evalNamesSet.contains(columnConfig.getColumnName())) {
                            featureNames.add(columnConfig.getColumnName());
                            outputNames.addAll(
                                    super.genNormColumnNames(columnConfig, this.modelConfig.getNormalizeType()));
                        } else {
                            throw new RuntimeException(varDesc + " - " + columnConfig.getColumnName()
                                    + " couldn't be found in eval dataset!");
                        }
                    } else { // expression appending variables, like abc_1, real column name is abc.
                        String realColumnName = columnConfig.getColumnName().substring(0,
                                columnConfig.getColumnName().lastIndexOf("_"));
                        if(evalNamesSet.contains(realColumnName)) {
                            featureNames.add(columnConfig.getColumnName());
                            outputNames.addAll(
                                    super.genNormColumnNames(columnConfig, this.modelConfig.getNormalizeType()));
                        } else {
                            throw new RuntimeException(varDesc + " - " + columnConfig.getColumnName()
                                    + " couldn't be found in eval dataset!");
                        }
                    }
                } else {
                    if(evalNamesSet.contains(columnConfig.getColumnName())) {
                        featureNames.add(columnConfig.getColumnName());
                        outputNames.addAll(super.genNormColumnNames(columnConfig, this.modelConfig.getNormalizeType()));
                    } else {
                        throw new RuntimeException(
                                varDesc + " - " + columnConfig.getColumnName() + " couldn't be found in eval dataset!");
                    }
                }
            }
        }
    }

    private void setPrecisionType() {
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.precisionType = PrecisionType.of(UDFContext.getUDFContext().getJobConf()
                    .get(Constants.SHIFU_NORM_PRECISION_TYPE, PrecisionType.FLOAT32.toString()));
        } else {
            this.precisionType = PrecisionType
                    .of(Environment.getProperty(Constants.SHIFU_NORM_PRECISION_TYPE, PrecisionType.FLOAT32.toString()));
        }
        if(this.precisionType == null) {
            this.precisionType = PrecisionType.FLOAT32;
        }
        log.info("Precision type is set to: " + this.precisionType);
    }

    private void setCategoryMissingNormType() {
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.categoryMissingNormType = CategoryMissingNormType
                    .of(UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_NORM_CATEGORY_MISSING_NORM));
        } else {
            this.categoryMissingNormType = CategoryMissingNormType
                    .of(Environment.getProperty(Constants.SHIFU_NORM_CATEGORY_MISSING_NORM));
        }
        if(this.categoryMissingNormType == null) {
            this.categoryMissingNormType = CategoryMissingNormType.POSRATE;
        }
        log.info("'categoryMissingNormType' is set to: " + this.categoryMissingNormType);
    }

    public Tuple exec(Tuple input) throws IOException {
        if(isCsvFormat) {
            String firstCol = ((input.get(0) == null) ? "" : input.get(0).toString());
            if(this.headers[0].equals(CommonUtils.normColumnName(firstCol))) {
                // Column value == Column Header? It's the first line of file?
                // TODO what to do if the column value == column name? ...
                return null;
            }
        }

        if(this.modelRunner == null && this.isAppendScore) {
            // here to initialize modelRunner, this is moved from constructor to here to avoid OOM in client side.
            // UDF in pig client will be initialized to get some metadata issues
            @SuppressWarnings("deprecation")
            List<BasicML> models = ModelSpecLoaderUtils.loadBasicModels(modelConfig, evalConfig,
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

        List<String> outputRawList = new ArrayList<>();

        Tuple tuple = TupleFactory.getInstance().newTuple();
        for(int i = 0; i < this.featureNames.size(); i++) {
            String name = this.featureNames.get(i);
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
                List<Double> normVals = Normalizer.fullNormalize(columnConfig, raw,
                        this.modelConfig.getNormalizeStdDevCutOff(), this.modelConfig.getNormalizeType(),
                        this.categoryMissingNormType, this.categoricalIndexMap.get(columnConfig.getColumnNum()));
                if(this.isOutputRaw) { // add to raw list
                    outputRawList.add(raw);
                }
                for(Double normVal: normVals) {
                    tuple.append(getOutputValue(normVal, true));
                }
            }
        }

        if(this.isOutputRaw) {
            outputRawList.forEach(raw -> tuple.append(raw));
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
                name = CommonUtils.normColumnName(name);
                if(i < 2 + validMetaSize) {
                    // set target, weight and meta columns to string
                    tupleSchema.add(new FieldSchema(name, DataType.CHARARRAY));
                } else {
                    tupleSchema.add(new FieldSchema(name, getOutputType()));
                }
            }

            if(this.isOutputRaw) { // append raw variable
                List<FieldSchema> rawFieldSchemaList = new ArrayList<>();
                for(int i = 2 + validMetaSize; i < this.featureNames.size(); i++) {
                    String name = this.featureNames.get(i);
                    ColumnConfig columnConfig = this.columnConfigMap.get(name);
                    if(columnConfig.isNumerical()) {
                        rawFieldSchemaList.add(new FieldSchema(name + ORIG_POSTFIX, getOutputType()));
                    } else {
                        rawFieldSchemaList.add(new FieldSchema(name + ORIG_POSTFIX, DataType.CHARARRAY));
                    }
                }
                rawFieldSchemaList.forEach(fieldSchema -> tupleSchema.add(fieldSchema));
            }

            if(this.isAppendScore) {
                tupleSchema.add(new FieldSchema(StringUtils.isBlank(this.scoreName) // no score
                        ? "default_score"
                        : this.scoreName, DataType.DOUBLE));
            }
            return new Schema(new FieldSchema("EvalNorm", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

    /**
     * FLOAT7 is old with DecimalFormat, new one with FLOAT16, FLOAT32, DOUBLE64
     */
    private String getOutputValue(double value, boolean enablePrecision) {
        if(enablePrecision) {
            switch(this.precisionType) {
                case FLOAT7:
                    return NormalizeUDF.DECIMAL_FORMAT.format(value);
                case FLOAT16:
                    return "" + NormalizeUDF.toFloat(NormalizeUDF.fromFloat((float) value));
                case DOUBLE64:
                    return value + "";
                case FLOAT32:
                default:
                    return ((float) value) + "";
            }
        } else {
            return ((float) value) + "";
        }
    }

    private byte getOutputType() {
        switch(this.precisionType) {
            case FLOAT7:
            case FLOAT16:
            case FLOAT32:
                return DataType.FLOAT;
            case DOUBLE64:
            default:
                return DataType.DOUBLE;
        }
    }

}
