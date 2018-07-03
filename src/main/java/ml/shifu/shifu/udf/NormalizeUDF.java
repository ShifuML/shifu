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

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.WeightAmplifier;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.tools.pigstats.PigStatusReporter;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

/**
 * NormalizeUDF class normalize the training data for parquet format.
 */
public class NormalizeUDF extends AbstractTrainerUDF<Tuple> {

    private static final String POSRATE = "posrate";
    private static final int MAX_MISMATCH_CNT = 500;

    private List<Set<String>> tags;

    private Double cutoff;
    private NormType normType;
    private Expression weightExpr;
    private JexlContext weightContext;
    public static DecimalFormat DECIMAL_FORMAT = new DecimalFormat("#.######");
    private PrecisionType precisionType;

    private List<DataPurifier> dataPurifiers;

    private boolean isForExpressions = false;
    private boolean isCompactNorm = false;

    private int mismatchCnt = 0;

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

    public static enum PrecisionType {
        FLOAT7, FLOAT16, FLOAT32, DOUBLE64;

        public static PrecisionType of(String precisionType) {
            for(PrecisionType pt: PrecisionType.values()) {
                if(pt.toString().equalsIgnoreCase(precisionType)) {
                    return pt;
                }
            }
            return FLOAT32;
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

    /**
     * Output compact column list for #isCompactNorm, schema is: tag, meta columns, feature list, weight
     */
    private List<String> outputCompactColumns;

    /**
     * Like schema in {@link #outputCompactColumns}, here is size of first tag and meta columns for output schema
     */
    private int cntOfTargetAndMetaColumns;

    private boolean isLinearTarget = false;

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        this(source, pathModelConfig, pathColumnConfig, "false");
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

        boolean isColumnSelected = false;
        for(ColumnConfig config: columnConfigList) {
            if(config.isFinalSelect() && !isColumnSelected) {
                isColumnSelected = true;
            }

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

        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            this.isCompactNorm = Boolean.TRUE.toString().equalsIgnoreCase(UDFContext.getUDFContext().getJobConf()
                    .get(Constants.SHIFU_NORM_ONLY_SELECTED, Boolean.FALSE.toString()));
        } else {
            this.isCompactNorm = Boolean.TRUE.toString().equalsIgnoreCase(
                    Environment.getProperty(Constants.SHIFU_NORM_ONLY_SELECTED, Boolean.FALSE.toString()));
        }

        // check if has final-select column, then enable real compact norm if user set,
        // isCompact now only works in non-tree model norm output
        if(isColumnSelected && isCompactNorm) {
            isCompactNorm = true;
        }

        // store schema list with format: tag, meta column, selected feature list, weight int a list
        if(isCompactNorm) {
            outputCompactColumns = new ArrayList<String>();
            outputCompactColumns.add(normColumnName(CommonUtils.findTargetColumn(columnConfigList).getColumnName()));
            for(ColumnConfig config: columnConfigList) {
                if(config.isMeta() && !config.isTarget()) {
                    outputCompactColumns.add(normColumnName(config.getColumnName()));
                }
            }
            // set cnt for output schema reference
            cntOfTargetAndMetaColumns = outputCompactColumns.size();
            for(ColumnConfig config: columnConfigList) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    outputCompactColumns.add(normColumnName(config.getColumnName()));
                }
            }
        }

        this.isLinearTarget = CommonUtils.isLinearTarget(modelConfig, columnConfigList);

        setPrecisionType();
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
            this.categoryMissingNormType = CategoryMissingNormType.of(
                    UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_NORM_CATEGORY_MISSING_NORM, POSRATE));
        } else {
            this.categoryMissingNormType = CategoryMissingNormType
                    .of(Environment.getProperty(Constants.SHIFU_NORM_CATEGORY_MISSING_NORM, POSRATE));
        }
        if(this.categoryMissingNormType == null) {
            this.categoryMissingNormType = CategoryMissingNormType.POSRATE;
        }
        log.info("'categoryMissingNormType' is set to: " + this.categoryMissingNormType);
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
        if(!isLinearTarget && !super.tagSet.contains(rawTag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        // data sampling only for normalization, for data cleaning, shouldn't do data sampling
        if(!isLinearTarget && !this.isForClean) {
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

        Map<String, Object> compactVarMap = null;
        if(this.isCompactNorm) {
            compactVarMap = new HashMap<String, Object>();
        }

        if(!this.isForExpressions) {
            if ( input.size() != this.columnConfigList.size() ) {
                this.mismatchCnt++;
                log.error("the input size - " + input.size() + ", while column size - " + columnConfigList.size());
                this.mismatchCnt++;
                // Throw exceptions if the mismatch count is greater than MAX_MISMATCH_CNT,
                // this could make Shifu could skip some malformed data
                if(this.mismatchCnt > MAX_MISMATCH_CNT) {
                    throw new ShifuException(ShifuErrorCode.ERROR_NO_EQUAL_COLCONFIG);
                }
                return null;
            }

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
                        if(this.isCompactNorm) {
                            compactVarMap.put(normColumnName(config.getColumnName()), type);
                        } else {
                            tuple.append(type);
                        }
                    } else if(this.isLinearTarget) {
                        double tagValue = 0.0;
                        try {
                            tagValue = Double.parseDouble(rawTag);
                        } catch (Exception e) {
                            log.error("Tag - " + rawTag + " is invalid(not numerical). Skip record.");
                            // skip this line
                            return null;
                        }
                        if(this.isCompactNorm) {
                            compactVarMap.put(normColumnName(config.getColumnName()), tagValue);
                        } else {
                            tuple.append(tagValue);
                        }
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
                            warn("Invalid data! The target value is not listed - " + rawTag,
                                    WarnInNormalizeUDF.INVALID_TAG);
                            return null;
                        }
                        if(this.isCompactNorm) {
                            compactVarMap.put(normColumnName(config.getColumnName()), index);
                        } else {
                            tuple.append(index);
                        }
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

                        appendOutputValue(tuple, normVal, true);
                    }
                } else {
                    if(this.isCompactNorm) {
                        // only output features and target, weight in compact norm mode
                        if(!config.isMeta() && config.isFinalSelect()) {
                            // for multiple classification, binPosRate means rate of such category over all counts,
                            // reuse binPosRate for normalize
                            List<Double> normVals = Normalizer.normalize(config, val, cutoff, normType,
                                    this.categoryMissingNormType);
                            for(Double normVal: normVals) {
                                String formatVal = getOutputValue(normVal, true);
                                compactVarMap.put(normColumnName(config.getColumnName()), formatVal);
                            }
                        } else if(config.isMeta()) {
                            compactVarMap.put(normColumnName(config.getColumnName()), val);
                        } else {
                            // if is compact mode but such column is not final selected, should be empty, as only append
                            // target and finalSelect feature, no need append here so this code block is empty. TODO, do
                            // we need meta column?
                        }
                    } else {
                        // append normalize data. exclude data clean, for data cleaning, no need check good or bad
                        // candidate
                        // By zhanhu: fix bug - if the ForceSelect variables exists in candidates list,
                        //      it will cause variable fail to normalize
                        if(CommonUtils.isToNormVariable(config, super.hasCandidates, modelConfig.isRegression())) {
                            // for multiple classification, binPosRate means rate of such category over all counts,
                            // reuse binPosRate for normalize
                            List<Double> normVals = Normalizer.normalize(config, val, cutoff, normType,
                                    this.categoryMissingNormType);
                            for(Double normVal: normVals) {
                                appendOutputValue(tuple, normVal, true);
                            }
                        } else {
                            tuple.append(config.isMeta() ? val : null);
                        }
                    }
                }
            }
        } else {
            // for segment expansion variables
            int rawSize = input.size();
            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                int newIndex = i >= rawSize ? i % rawSize : i;
                String val = (input.get(newIndex) == null) ? "" : input.get(newIndex).toString().trim();

                // for target column
                if(config.isTarget()) {
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
                        if(this.isCompactNorm) {
                            compactVarMap.put(normColumnName(config.getColumnName()), type);
                        } else {
                            tuple.append(type);
                        }
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
                            warn("Invalid data! The target value is not listed - " + rawTag,
                                    WarnInNormalizeUDF.INVALID_TAG);
                            return null;
                        }
                        if(this.isCompactNorm) {
                            compactVarMap.put(normColumnName(config.getColumnName()), index);
                        } else {
                            tuple.append(index);
                        }
                    }
                    continue;
                }

                if(this.isCompactNorm) {
                    // only output features and target, weight in compact norm mode
                    if(!config.isMeta() && config.isFinalSelect()) {
                        // for multiple classification, binPosRate means rate of such category over all counts,
                        // reuse binPosRate for normalize
                        List<Double> normVals = Normalizer.normalize(config, val, cutoff, normType,
                                this.categoryMissingNormType);
                        for(Double normVal: normVals) {
                            String formatVal = getOutputValue(normVal, true);
                            compactVarMap.put(normColumnName(config.getColumnName()), formatVal);
                        }
                    } else if(config.isMeta()) {
                        compactVarMap.put(normColumnName(config.getColumnName()), val);
                    } else {
                        // if is compact mode but such column is not final selected, should be empty, as only append
                        // target and finalSelect feature, no need append here so this code block is empty. TODO, do
                        // we need meta column?
                    }
                } else {
                    // for others
                    if(CommonUtils.isToNormVariable(config, super.hasCandidates, modelConfig.isRegression())) {
                        List<Double> normVals = Normalizer.normalize(config, val, cutoff, normType,
                                this.categoryMissingNormType);
                        for(Double normVal: normVals) {
                            appendOutputValue(tuple, normVal, true);
                        }
                    } else {
                        tuple.append(config.isMeta() ? val : null);
                    }
                }

            }
        }

        // for compact norm mode, output to tuple at here
        if(this.isCompactNorm) {
            for(int i = 0; i < outputCompactColumns.size(); i++) {
                tuple.append(compactVarMap.get(outputCompactColumns.get(i)));
            }
        }

        // append tuple with weight.
        double weight = evaluateWeight(weightExpr, weightContext);
        tuple.append(weight);

        return tuple;
    }

    /**
     * FLOAT7 is old with DecimalFormat, new one with FLOAT16, FLOAT32, DOUBLE64
     */
    private String getOutputValue(double value, boolean enablePrecision) {
        if(enablePrecision) {
            switch(this.getPrecisionType()) {
                case FLOAT7:
                    return DECIMAL_FORMAT.format(value);
                case FLOAT16:
                    return "" + toFloat(fromFloat((float) value));
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

    /**
     * FLOAT7 is old with DecimalFormat, new one with FLOAT16, FLOAT32, DOUBLE64
     */
    private void appendOutputValue(Tuple tuple, double value, boolean enablePrecision) {
        if(enablePrecision) {
            switch(this.getPrecisionType()) {
                case FLOAT7:
                    tuple.append(DECIMAL_FORMAT.format(value));
                    break;
                case FLOAT16:
                    tuple.append(toFloat(fromFloat((float) value)));
                    break;
                case DOUBLE64:
                    tuple.append(value);
                    break;
                case FLOAT32:
                default:
                    tuple.append((float) value);
                    break;
            }
        } else {
            tuple.append((float) value);
        }
    }

    // returns all higher 16 bits as 0 for all results
    public static int fromFloat(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = fbits >>> 16 & 0x8000; // sign only
        int val = (fbits & 0x7fffffff) + 0x1000; // rounded value

        if(val >= 0x47800000) // might be or become NaN/Inf
        { // avoid Inf due to rounding
            if((fbits & 0x7fffffff) >= 0x47800000) { // is or must become NaN/Inf
                if(val < 0x7f800000) // was value but too large
                    return sign | 0x7c00; // make it +/-Inf
                return sign | 0x7c00 | // remains +/-Inf or NaN
                        (fbits & 0x007fffff) >>> 13; // keep NaN (and Inf) bits
            }
            return sign | 0x7bff; // unrounded not quite Inf
        }
        if(val >= 0x38800000) // remains normalized value
            return sign | val - 0x38000000 >>> 13; // exp - 127 + 15
        if(val < 0x33000000) // too small for subnormal
            return sign; // becomes +/-0
        val = (fbits & 0x7fffffff) >>> 23; // tmp exp for subnormal calc
        return sign | ((fbits & 0x7fffff | 0x800000) // add subnormal bit
                + (0x800000 >>> val - 102) // round depending on cut off
        >>> 126 - val); // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
    }

    // ignores the higher 16 bits
    public static float toFloat(int hbits) {
        int mant = hbits & 0x03ff; // 10 bits mantissa
        int exp = hbits & 0x7c00; // 5 bits exponent
        if(exp == 0x7c00) // NaN/Inf
            exp = 0x3fc00; // -> NaN/Inf
        else if(exp != 0) // normalized value
        {
            exp += 0x1c000; // exp - 15 + 127
            if(mant == 0 && exp > 0x1c400) // smooth transition
                return Float.intBitsToFloat((hbits & 0x8000) << 16 | exp << 13 | 0x3ff);
        } else if(mant != 0) // && exp==0 -> subnormal
        {
            exp = 0x1c400; // make it normal
            do {
                mant <<= 1; // mantissa * 2
                exp -= 0x400; // decrease exp by 1
            } while((mant & 0x400) == 0); // while not normal
            mant &= 0x3ff; // discard subnormal bit
        } // else +/-0 -> +/-0
        return Float.intBitsToFloat( // combine all parts
                (hbits & 0x8000) << 16 // sign << ( 31 - 15 )
                        | (exp | mant) << 13); // value << ( 23 - 10 )
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
            Schema tupleSchema = null;
            if(this.isCompactNorm) {
                // compact norm, no need to Normalized schema
                tupleSchema = new Schema();
            } else {
                schemaStr.append("Normalized:Tuple(");
            }

            if(this.isCompactNorm) {
                // compact norm mode, schema is tag, meta columns, feature columns and weight
                for(int i = 0; i < outputCompactColumns.size(); i++) {
                    String normName = normColumnName(outputCompactColumns.get(i));
                    if(i == 0) {
                        // target column
                        tupleSchema.add(new Schema.FieldSchema(normName, DataType.DOUBLE));
                    } else if(i < cntOfTargetAndMetaColumns) {
                        // meta column
                        tupleSchema.add(new Schema.FieldSchema(normName, DataType.CHARARRAY));
                    } else {
                        // feature column
                        switch(this.getPrecisionType()) {
                            case DOUBLE64:
                                tupleSchema.add(new Schema.FieldSchema(normName, DataType.DOUBLE));
                            case FLOAT7:
                            case FLOAT16:
                            case FLOAT32:
                            default:
                                // all these types are actually float in Java/Pig
                                tupleSchema.add(new Schema.FieldSchema(normName, DataType.FLOAT));
                        }
                    }
                }
            } else {
                for(ColumnConfig config: columnConfigList) {
                    if(config.isMeta()) {
                        schemaStr.append(normColumnName(config.getColumnName()) + ":chararray" + ",");
                    } else if(config.isTarget()) {
                        schemaStr.append(normColumnName(config.getColumnName()) + ":double" + ",");
                    } else if(config.isNumerical()) {
                        if(CommonUtils.isToNormVariable(config, super.hasCandidates, modelConfig.isRegression())) {
                            if(modelConfig.getNormalizeType().equals(NormType.ONEHOT)) {
                                // one hot logic
                                for(int i = 0; i < config.getBinBoundary().size(); i++) {
                                    schemaStr.append(normColumnName(config.getColumnName()) + "_" + i + ":"
                                            + getOutputPrecisionType() + ",");
                                }
                                schemaStr.append(normColumnName(config.getColumnName()) + "_missing" + ":"
                                        + getOutputPrecisionType() + ",");
                            } else {
                                // non one hot
                                schemaStr.append(
                                        normColumnName(config.getColumnName()) + ":" + getOutputPrecisionType() + ",");
                            }
                        } else {
                            // other not good candidate
                            schemaStr.append(
                                    normColumnName(config.getColumnName()) + ":" + getOutputPrecisionType() + ",");
                        }
                    } else {
                        if(config.isCategorical() && this.isForClean) {
                            // clean data for DT algorithms, only store index, short is ok while Pig only have int
                            // type
                            schemaStr.append(normColumnName(config.getColumnName()) + ":chararray" + ",");
                        } else {
                            // for others, set to float, no matter LR/NN categorical or filter out feature with null
                            if(CommonUtils.isToNormVariable(config, super.hasCandidates, modelConfig.isRegression())) {
                                if(modelConfig.getNormalizeType().equals(NormType.ZSCALE_ONEHOT)
                                        || modelConfig.getNormalizeType().equals(NormType.ONEHOT)) {
                                    // one hot logic
                                    for(int i = 0; i < config.getBinCategory().size(); i++) {
                                        schemaStr.append(normColumnName(config.getColumnName()) + "_" + i + ":"
                                                + getOutputPrecisionType() + ",");
                                    }
                                    schemaStr.append(normColumnName(config.getColumnName()) + "_missing" + ":"
                                            + getOutputPrecisionType() + ",");
                                } else {
                                    // non one hot
                                    schemaStr.append(normColumnName(config.getColumnName()) + ":"
                                            + getOutputPrecisionType() + ",");
                                }
                            } else {
                                // other not good candidate
                                schemaStr.append(
                                        normColumnName(config.getColumnName()) + ":" + getOutputPrecisionType() + ",");
                            }
                        }
                    }
                }
            }

            if(this.isCompactNorm) {
                switch(this.getPrecisionType()) {
                    case DOUBLE64:
                        tupleSchema.add(new Schema.FieldSchema("weight", DataType.DOUBLE));
                    case FLOAT7:
                    case FLOAT16:
                    case FLOAT32:
                    default:
                        // all these types are actually float in Java/Pig
                        tupleSchema.add(new Schema.FieldSchema("weight", DataType.FLOAT));
                }
                // TODO, Even set null schema alias, final pig_header output still has null::weight or ::weight, we
                // would like to have weight only.
                Schema schema = new Schema(new Schema.FieldSchema("Normalized", tupleSchema, DataType.TUPLE));
                return schema;
            } else {
                schemaStr.append("weight:").append(getOutputPrecisionType()).append(")");
                return Utils.getSchemaFromString(schemaStr.toString());
            }
        } catch (Exception e) {
            log.error("error in outputSchema", e);
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

    private String getOutputPrecisionType() {
        switch(this.getPrecisionType()) {
            case DOUBLE64:
                return "double";
            case FLOAT7:
            case FLOAT16:
            case FLOAT32:
                // all these types are actually
            default:
                return "float";
        }
    }

    public PrecisionType getPrecisionType() {
        return precisionType;
    }

    public void setPrecisionType(PrecisionType precisionType) {
        this.precisionType = precisionType;
    }
}
