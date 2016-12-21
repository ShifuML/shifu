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

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.WeightAmplifier;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.tools.pigstats.PigStatusReporter;

/**
 * NormalizeUDF class normalize the training data for parquet format.
 */
public class NormalizeUDF extends AbstractTrainerUDF<Tuple> {

    private List<Set<String>> tags;

    private Double cutoff;
    private NormType normType;
    private Expression weightExpr;
    private JexlContext weightContext;
    private DecimalFormat df = new DecimalFormat("#.######");

    public static enum WarnInNormalizeUDF {
        INVALID_TAG;
    };

    // if current norm for only clean and not transform categorical and numeric value
    private boolean isForClean = false;

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        this(source, pathModelConfig, pathColumnConfig, "false");
    }

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig, String isForClean)
            throws Exception {
        super(source, pathModelConfig, pathColumnConfig);
        this.isForClean = "true".equalsIgnoreCase(isForClean);

        log.debug("Initializing NormalizeUDF ... ");

        cutoff = modelConfig.getNormalizeStdDevCutOff();
        log.debug("\t stdDevCutOff: " + cutoff);

        normType = modelConfig.getNormalizeType();
        log.debug("\t normType: " + normType.name());

        weightExpr = createExpression(modelConfig.getWeightColumnName());
        if(weightExpr != null) {
            weightContext = new MapContext();
        }

        this.tags = super.modelConfig.getSetTags();

        log.debug("NormalizeUDF Initialized");
    }

    @SuppressWarnings("deprecation")
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() == 0) {
            return null;
        }

        final String rawTag = input.get(tagColumnNum).toString();

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

        for(int i = 0; i < input.size(); i++) {
            ColumnConfig config = columnConfigList.get(i);
            String val = (input.get(i) == null) ? "" : input.get(i).toString().trim();
            // load variables for weight calculating.
            if(weightExpr != null) {
                weightContext.set(config.getColumnName(), val);
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
                        warn("Invalid data! The target value is not listed - " + rawTag, WarnInNormalizeUDF.INVALID_TAG);
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

            // append normalize data.
            if(!CommonUtils.isGoodCandidate(modelConfig.isRegression(), config)) {
                if(config.isMeta()) {
                    tuple.append(val);
                } else {
                    tuple.append(null);
                }
            } else {
                if(this.isForClean) {
                    // for RF/GBT model, only clean data, not real do norm data
                    if(config.isCategorical()) {
                        // TODO using HashSet instead of ArrayList
                        int index = config.getBinCategory().indexOf(val);
                        if(index == -1) {
                            // set to empty for invalid category
                            tuple.append("");
                        } else {
                            tuple.append(val);
                        }
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
                    // for multiple classification, binPosRate means rate of such category over all counts, reuse
                    // binPosRate for normalize
                    Double normVal = Normalizer.normalize(config, val, cutoff, normType);
                    tuple.append(df.format(normVal));
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
                        schemaStr.append(config.getColumnName() + ":float" + ",");
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

    /**
     * Create expressions for multi weight settings
     * 
     * @param weightExprList
     * @return weight expression map
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

    /**
     * Create the expression for weight setting
     * 
     * @param weightAmplifier
     * @return expression for weight amplifier
     */
    private Expression createExpression(String weightAmplifier) {
        if(StringUtils.isNotBlank(weightAmplifier)) {
            JexlEngine jexl = new JexlEngine();
            return jexl.createExpression(weightAmplifier);
        }
        return null;
    }
}
