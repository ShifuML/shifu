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
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.WeightAmplifier;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;

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

/**
 * NormalizeUDF class normalize the training data
 */
public class NormalizeUDF extends AbstractTrainerUDF<Tuple> {

    private List<String> negTags;
    private List<String> posTags;
    private Expression weightExpr;
    private DecimalFormat df = new DecimalFormat("#.######");

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        log.debug("Initializing NormalizeUDF ... ");

        negTags = modelConfig.getNegTags();
        log.debug("\t Negative Tags: " + negTags);
        posTags = modelConfig.getPosTags();
        log.debug("\t Positive Tags: " + posTags);

        weightExpr = createExpression(modelConfig.getWeightColumnName());
        log.debug("NormalizeUDF Initialized");
    }

    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() == 0) {
            return null;
        }

        int size = input.size();

        // new ????
        JexlContext jc = new MapContext();

        Tuple tuple = TupleFactory.getInstance().newTuple();

        String tag = input.get(tagColumnNum).toString();
        if(!(posTags.contains(tag) || negTags.contains(tag))) {
            // avoid too many logs
            if(System.currentTimeMillis() % 100 == 0) {
                log.warn("Invalid target column value - " + tag);
            }
            return null;
        }

        boolean isNotSampled = DataSampler.isNotSampled(modelConfig.getPosTags(), modelConfig.getNegTags(),
                modelConfig.getNormalizeSampleRate(), modelConfig.isNormalizeSampleNegOnly(), tag);
        if(isNotSampled) {
            return null;
        }

        Double cutoff = modelConfig.getNormalizeStdDevCutOff();
        for(int i = 0; i < size; i++) {
            ColumnConfig config = columnConfigList.get(i);
            if(weightExpr != null) {
                jc.set(config.getColumnName(), ((input.get(i) == null) ? "" : input.get(i).toString()));
            }

            if(super.tagColumnNum == i) {
                if(modelConfig.getPosTags().contains(tag)) {
                    tuple.append(df.format(Double.valueOf(1)));
                } else if(modelConfig.getNegTags().contains(tag)) {
                    tuple.append(df.format(Double.valueOf(0)));
                } else {
                    log.error("Invalid data! The target value is not listed - " + tag);
                    // Return null to skip such record.
                    return null;
                }
                continue;
            }

            if(!CommonUtils.isGoodCandidate(config)) {
                tuple.append(null);
            } else {
                String val = ((input.get(i) == null) ? "" : input.get(i).toString());
                Double z = Normalizer.normalize(config, val, cutoff);
                tuple.append(df.format(z));
            }
        }

        double weight = 1.0d;
        if(weightExpr != null) {
            Object result = weightExpr.evaluate(jc);
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
        tuple.append(weight);

        return tuple;
    }

    public Schema outputSchema(Schema input) {
        try {
            StringBuilder schemaStr = new StringBuilder();
            schemaStr.append("Normalized:Tuple(");
            for(ColumnConfig config: columnConfigList) {
                if(!config.isMeta() && config.isNumerical()) {
                    schemaStr.append(config.getColumnName() + ":float" + ",");
                } else {
                    schemaStr.append(config.getColumnName() + ":chararray" + ",");
                }
            }
            schemaStr.append("weight:float)");

            return Utils.getSchemaFromString(schemaStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
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
                ewMap.put(jexl.createExpression(we.getTargetExpression()), Double.valueOf(we.getTargetWeight()));
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
