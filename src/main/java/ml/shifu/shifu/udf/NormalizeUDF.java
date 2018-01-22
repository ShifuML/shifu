/*
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
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.WeightAmplifier;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.tools.pigstats.PigStatusReporter;

/**
 * NormalizeUDF class normalize the training data for parquet format.
 */
public class NormalizeUDF extends AbstractTrainerUDF<Tuple> {

    private static final String POSRATE = "posrate";

    private List<Set<String>> tags;

    private Double cutoff;
    private NormType normType;
    private Expression weightExpr;
    private JexlContext weightContext;
    private DecimalFormat df = new DecimalFormat("#.######");

    private List<DataPurifier> dataPurifiers;

    private boolean isForExpressions = false;

    public static enum CategoryMissingNormType {
        MEAN, POSRATE;

        public static CategoryMissingNormType of(String normType) {
            for(CategoryMissingNormType norm: CategoryMissingNormType.values()) {
                if(norm.toString().equalsIgnoreCase(normType)) {
                    return norm;
                }
            }
            return POSRATE;
        }
    }

    /**
     * For categorical feature, a map is used to save query time in execution
     */
    private Map<Integer, Map<String, Integer>> categoricalIndexMap = new HashMap<Integer, Map<String, Integer>>();

    public static enum WarnInNormalizeUDF {
        INVALID_TAG;
    };

    // if current norm for only clean and not transform categorical and numeric value
    private boolean isForClean = false;

    /**
     * In Zscore norm type, how to process category default missing value norm, by default use mean, another option is
     * POSRATE.
     */
    private CategoryMissingNormType categoryMissingNormType = CategoryMissingNormType.POSRATE;

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        this(source, pathModelConfig, pathColumnConfig, "false");
        setCategoryMissingNormType();
    }

    private void setCategoryMissingNormType() {
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.categoryMissingNormType = CategoryMissingNormType.of(UDFContext.getUDFContext().getJobConf()
                    .get(Constants.SHIFU_NORM_CATEGORY_MISSING_NORM, POSRATE));
        } else {
            this.categoryMissingNormType = CategoryMissingNormType.of(Environment.getProperty(
                    Constants.SHIFU_NORM_CATEGORY_MISSING_NORM, POSRATE));
        }
        if(this.categoryMissingNormType == null) {
            this.categoryMissingNormType = CategoryMissingNormType.POSRATE;
        }
        log.info("'categoryMissingNormType' is set to: " + this.categoryMissingNormType);
    }

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig, String isForClean)
            throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        setCategoryMissingNormType();

        this.isForClean = "true".equalsIgnoreCase(isForClean);

        cutoff = modelConfig.getNormalizeStdDevCutOff();

        normType = modelConfig.getNormalizeType();
        log.debug("\t normType: " + normType.name());

        weightExpr = createExpression(modelConfig.getWeightColumnName());
        if(weightExpr != null) {
            weightContext = new MapContext();
        }

        this.tags = super.modelConfig.getSetTags();

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

        String filterExpressions = "";

        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            filterExpressions = UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_SEGMENT_EXPRESSIONS);
        } else {
            filterExpressions = Environment.getProperty(Constants.SHIFU_SEGMENT_EXPRESSIONS);
        }

        if(StringUtils.isNotBlank(filterExpressions)) {
            this.isForExpressions = true;
            String[] splits = CommonUtils.split(filterExpressions, Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
            this.dataPurifiers = new ArrayList<DataPurifier>(splits.length);
            for(String split: splits) {
                this.dataPurifiers.add(new DataPurifier(modelConfig, split, false));
            }
        }

    }

    @SuppressWarnings("deprecation")
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() == 0) {
            return null;
        }

        Object tag = input.get(tagColumnNum);
        if(tag == null) {
            log.warn("The tag is NULL, just skip it!!");
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }
        final String rawTag = CommonUtils.trimTag(tag.toString());

        // make sure all invalid tag record are filter out
        if(!super.tagSet.contains(rawTag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        // data sampling only for normalization, for data cleaning, shouldn't do data sampling
        if(!this.isForClean) {
            // do data sampling. Unselected data or data with invalid tag will be filtered out.
            boolean isNotSampled = DataSampler.isNotSampled(modelConfig.isRegression(), super.tagSet, super.posTagSet,
                    super.negTagSet, modelConfig.getNormalizeSampleRate(), modelConfig.isNormalizeSampleNegOnly(),
                    rawTag);
            if(isNotSampled) {
                return null;
            }
        }

        // append tuple with tag, normalized value.
        Tuple tuple = TupleFactory.getInstance().newTuple();
        final NormType normType = modelConfig.getNormalizeType();

        if(!this.isForExpressions) {
            for(int i = 0; i < input.size(); i++) {
                ColumnConfig config = columnConfigList.get(i);
                String val = (input.get(i) == null) ? "" : input.get(i).toString().trim();
                // load variables for weight calculating.
                if(weightExpr != null) {
                    weightContext.set(new NSColumn(config.getColumnName()).getSimpleName(), val);
                }

                // check tag type.
                if(tagColumnNum == i) {
                    if(modelConfig.isRegression()) {
                        int type = 0;
                        if(super.posTagSet.contains(rawTag)) {
                            type = 1;
                        } else if(super.negTagSet.contains(rawTag)) {
                            type = 0;
                        } else {
                            log.error("Invalid data! The target value is not listed - " + rawTag);
                            warn("Invalid data! The target value is not listed - " + rawTag,
                                    WarnInNormalizeUDF.INVALID_TAG);
                            return null;
                        }
                        tuple.append(type);
                    } else {
                        int index = -1;
                        for(int j = 0; j < tags.size(); j++) {
                            Set<String> tagSet = tags.get(j);
                            if(tagSet.contains(rawTag)) {
                                index = j;
                                break;
                            }
                        }
                        if(index == -1) {
                            log.error("Invalid data! The target value is not listed - " + rawTag);
                            return null;
                        }
                        tuple.append(index);
                    }
                    continue;
                }

                if(this.isForClean) {
                    // for RF/GBT model, only clean data, not real do norm data
                    if(config.isCategorical()) {
                        Map<String, Integer> map = this.categoricalIndexMap.get(config.getColumnNum());
                        // map should not be null, no need check if map is null, if val not in binCategory, set it to ""
                        tuple.append(((map.get(val) == null || map.get(val) == -1)) ? "" : val);
                    } else {
                        Double normVal = 0d;
                        try {
                            normVal = Double.parseDouble(val);
                        } catch (Exception e) {
                            log.debug("Not decimal format " + val + ", using default!");
                            normVal = Normalizer.defaultMissingValue(config);
                        }
                        tuple.append(df.format(normVal));
                    }
                } else {
                    // append normalize data. exclude data clean, for data cleaning, no need check good or bad candidate
                    if(CommonUtils.isGoodCandidate(config, super.hasCandidates, modelConfig.isRegression())) {
                        // for multiple classification, binPosRate means rate of such category over all counts, reuse
                        // binPosRate for normalize
                        List<Double> normVals = Normalizer.normalize(config, val, cutoff, normType,
                                this.categoryMissingNormType);
                        for(Double normVal: normVals) {
                            tuple.append(df.format(normVal));
                        }
                    } else {
                        tuple.append(config.isMeta() ? val : null);
                    }
                }
            }
        } else {
            int rawSize = input.size();
            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                int newIndex = i >= rawSize ? i % rawSize : i;
                String val = (input.get(newIndex) == null) ? "" : input.get(newIndex).toString().trim();
                if(config.isTarget()) {
                    int type = 0;
                    if(super.posTagSet.contains(rawTag)) {
                        type = 1;
                    } else if(super.negTagSet.contains(rawTag)) {
                        type = 0;
                    } else {
                        log.error("Invalid data! The target value is not listed - " + rawTag);
                        warn("Invalid data! The target value is not listed - " + rawTag, WarnInNormalizeUDF.INVALID_TAG);
                        return null;
                    }
                    tuple.append(type);
                } else if(CommonUtils.isGoodCandidate(config, super.hasCandidates, modelConfig.isRegression())) {
                    List<Double> normVals = Normalizer.normalize(config, val, cutoff, normType,
                            this.categoryMissingNormType);
                    for(Double normVal: normVals) {
                        tuple.append(df.format(normVal));
                    }
                } else {
                    tuple.append(config.isMeta() ? val : null);
                }
            }
        }

        // append tuple with weight.
        double weight = evaluateWeight(weightExpr, weightContext);
        tuple.append(weight);

        return tuple;
    }

    /**
     * Evaluate weight expression based on the variables context.
     * 
     * @param expr
     *            - weight evaluation expression
     * @param jc
     *            - A JexlContext containing variables for weight expression.
     * @return The result of this evaluation
     */
    public double evaluateWeight(Expression expr, JexlContext jc) {
        double weight = 1.0d;
        if(expr != null) {
            Object result = expr.evaluate(jc);
            if(result instanceof Integer) {
                weight = ((Integer) result).doubleValue();
            } else if(result instanceof Double) {
                weight = ((Double) result).doubleValue();
            } else if(result instanceof String) {
                try {
                    weight = Double.parseDouble((String) result);
                } catch (NumberFormatException e) {
                    // Not a number, use default
                    if(System.currentTimeMillis() % 100 == 0) {
                        log.warn("Weight column type is String and value cannot be parsed with " + result
                                + ", use default 1.0d");
                    }
                    weight = 1.0d;
                }
            }
        }
        return weight;
    }

    public Schema outputSchema(Schema input) {
        try {
            StringBuilder schemaStr = new StringBuilder();
            schemaStr.append("Normalized:Tuple(");
            for(ColumnConfig config: columnConfigList) {
                if(config.isMeta()) {
                    schemaStr.append(config.getColumnName() + ":chararray" + ",");
                } else if(!config.isMeta() && config.isNumerical()) {
                    schemaStr.append(config.getColumnName() + ":float" + ",");
                } else if(config.isTarget()) {
                    schemaStr.append(config.getColumnName() + ":int" + ",");
                } else {
                    if(config.isCategorical() && this.isForClean) {
                        // clean data for DT algorithms, only store index, short is ok while Pig only have int type
                        schemaStr.append(config.getColumnName() + ":chararray" + ",");
                    } else {
                        // for others, set to float, no matter LR/NN categorical or filter out feature with null
                        if(modelConfig.getNormalizeType().equals(NormType.ZSCALE_ONEHOT)) {
                            if(CommonUtils.isGoodCandidate(config, super.hasCandidates)) {
                                for(int i = 0; i < config.getBinCategory().size(); i++) {
                                    schemaStr.append(config.getColumnName() + "_" + i + ":float" + ",");
                                }
                            }
                            schemaStr.append(config.getColumnName() + "_missing" + ":float" + ",");
                        } else {
                            schemaStr.append(config.getColumnName() + ":float" + ",");
                        }
                    }
                }
            }
            schemaStr.append("weight:float)");
            return Utils.getSchemaFromString(schemaStr.toString());
        } catch (Exception e) {
            log.error("error in outputSchema", e);
            return null;
        }
    }

    /*
     * Create expressions for multi weight settings
     */
    protected Map<Expression, Double> createExpressionMap(List<WeightAmplifier> weightExprList) {
        Map<Expression, Double> ewMap = new HashMap<Expression, Double>();

        if(CollectionUtils.isNotEmpty(weightExprList)) {
            JexlEngine jexl = new JexlEngine();

            for(WeightAmplifier we: weightExprList) {
                ewMap.put(jexl.createExpression(we.getTargetExpression()), we.getTargetWeight());
            }
        }

        return ewMap;
    }

    /*
     * Create the expression for weight setting
     */
    private Expression createExpression(String weightAmplifier) {
        if(StringUtils.isNotBlank(weightAmplifier)) {
            JexlEngine jexl = new JexlEngine();
            return jexl.createExpression(weightAmplifier);
        }
        return null;
    }
}
