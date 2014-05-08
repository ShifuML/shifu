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
package ml.shifu.shifu.actor.worker;

import akka.actor.ActorRef;
import ml.shifu.shifu.container.WeightAmplifier;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.message.NormDataPrepMessage;
import ml.shifu.shifu.message.NormResultDataMessage;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * DataNormalizeWorker class is to normalize the train data
 * Notice, the last field of normalized data is the weight of the training data.
 * The weight is set in @ModelConfig.normalize.weightAmplifier. It could be some column
 */
public class DataNormalizeWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(DataNormalizeWorker.class);
    private Expression weightExpr;

    public DataNormalizeWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        weightExpr = createExpression(modelConfig.getWeightColumnName());
    }

    /* (non-Javadoc)
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) {
        if (message instanceof NormDataPrepMessage) {
            NormDataPrepMessage msg = (NormDataPrepMessage) message;
            List<String[]> rfList = msg.getRfList();
            int targetMsgCnt = msg.getTotalMsgCnt();

            List<List<Double>> normalizedDataList = normalizeData(rfList);
            nextActorRef.tell(new NormResultDataMessage(targetMsgCnt, rfList, normalizedDataList), this.getSelf());
        } else {
            unhandled(message);
        }
    }

    /**
     * Normalize the list training data from List<String> to List<Double>
     *
     * @param rfList
     * @return the data after normalization
     */
    private List<List<Double>> normalizeData(List<String[]> rfList) {
        List<List<Double>> normalizedDataList = new ArrayList<List<Double>>();

        for (String[] rf : rfList) {
            List<Double> normRecord = normalizeRecord(rf);
            if (CollectionUtils.isNotEmpty(normRecord)) {
                normalizedDataList.add(normRecord);
            }
        }

        return normalizedDataList;
    }

    /**
     * Normalize the training data record
     *
     * @param rfs - record fields
     * @return the data after normalization
     */
    private List<Double> normalizeRecord(String[] rfs) {
        List<Double> retDouList = new ArrayList<Double>();

        if (rfs == null || rfs.length == 0) {
            return null;
        }

        String tag = rfs[this.targetColumnNum];
        
        boolean isNotSampled = DataSampler.isNotSampled(
                modelConfig.getPosTags(), 
                modelConfig.getNegTags(),
                modelConfig.getNormalizeSampleRate(), 
                modelConfig.isNormalizeSampleNegOnly(), tag);
        if ( isNotSampled ) {
            return null;
        }
        
        if (modelConfig.getPosTags().contains(tag)) {
            retDouList.add(Double.valueOf(1));
        } else if (modelConfig.getNegTags().contains(tag)) {
            retDouList.add(Double.valueOf(0));
        } else {
            log.error("Invalid data! The target value is not listed - " + tag);
        }

        JexlContext jc = new MapContext();
        Double cutoff = modelConfig.getNormalizeStdDevCutOff();

        for (int i = 0; i < rfs.length; i++) {
            ColumnConfig config = columnConfigList.get(i);
            if (weightExpr != null) {
                jc.set(config.getColumnName(), rfs[i]);
            }

            if (config.isFinalSelect()) {
                String val = (rfs[i] == null) ? "" : rfs[i];
                Double z = Normalizer.normalize(config, val, cutoff);
                retDouList.add(z);
            }
        }

        double weight = 1.0d;
        if (weightExpr != null) {
            Object result = weightExpr.evaluate(jc);
            if (result instanceof Integer) {
                weight = ((Integer) result).doubleValue();
            } else if (result instanceof Double) {
                weight = ((Double) result).doubleValue();
            }
        }
        retDouList.add(weight);

        return retDouList;
    }

    /**
     * Create expressions for multi weight settings
     *
     * @param weightExprList
     * @return weight expression map
     */
    protected Map<Expression, Double> createExpressionMap(List<WeightAmplifier> weightExprList) {
        Map<Expression, Double> ewMap = new HashMap<Expression, Double>();

        if (CollectionUtils.isNotEmpty(weightExprList)) {
            JexlEngine jexl = new JexlEngine();

            for (WeightAmplifier we : weightExprList) {
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
        if (StringUtils.isNotBlank(weightAmplifier)) {
            JexlEngine jexl = new JexlEngine();
            return jexl.createExpression(weightAmplifier);
        }
        return null;
    }
}
