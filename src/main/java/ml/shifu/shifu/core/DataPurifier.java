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
package ml.shifu.shifu.core;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * DataPurifier class
 */
public class DataPurifier {

    private static Logger log = LoggerFactory.getLogger(DataPurifier.class);

    private String[] headers;
    private String dataDelimiter;
    private Expression dataFilterExpr;
    private ShifuMapContext jc = new ShifuMapContext();

    public DataPurifier(ModelConfig modelConfig) throws IOException {
        if(StringUtils.isNotBlank(modelConfig.getFilterExpressions())) {
            JexlEngine jexl = new JexlEngine();
            try {
                dataFilterExpr = jexl.createExpression(modelConfig.getFilterExpressions());
            } catch (JexlException e) {
                log.error("The expression is {} is invalid, please use correct expression.",
                        modelConfig.getFilterExpressions());
                dataFilterExpr = null;
            }
            this.headers = CommonUtils.getFinalHeaders(modelConfig);
            dataDelimiter = modelConfig.getDataSetDelimiter();
        }
    }
    
    public DataPurifier(ModelConfig modelConfig, String filterExpressions) throws IOException {
        if(StringUtils.isNotBlank(filterExpressions)) {
            JexlEngine jexl = new JexlEngine();
            try {
                dataFilterExpr = jexl.createExpression(filterExpressions);
            } catch (JexlException e) {
                log.error("The expression is {} is invalid, please use correct expression.",
                        modelConfig.getFilterExpressions());
                dataFilterExpr = null;
            }
            this.headers = CommonUtils.getFinalHeaders(modelConfig);
            dataDelimiter = modelConfig.getDataSetDelimiter();
        }
    }

    public DataPurifier(EvalConfig evalConfig) throws IOException {
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getFilterExpressions())) {
            JexlEngine jexl = new JexlEngine();
            try {
                dataFilterExpr = jexl.createExpression(evalConfig.getDataSet().getFilterExpressions());
            } catch (JexlException e) {
                log.error("The expression is {} is invalid, please use correct expression.", evalConfig.getDataSet()
                        .getFilterExpressions());
                dataFilterExpr = null;
            }

            headers = CommonUtils.getFinalHeaders(evalConfig);
            dataDelimiter = evalConfig.getDataSet().getDataDelimiter();
        }
    }

    public Boolean isFilter(String record) {
        if(dataFilterExpr == null) {
            return true;
        }

        String[] fields = CommonUtils.split(record, dataDelimiter);
        if(fields == null || fields.length != headers.length) {
            // illegal format data, just skip
            return false;
        }

        jc.clear();

        for(int i = 0; i < fields.length; i++) {
            NSColumn nsColumn = new NSColumn(headers[i]);
            jc.set(headers[i], ((fields[i] == null) ? "" : fields[i].toString()));
            jc.set(nsColumn.getSimpleName(), ((fields[i] == null) ? "" : fields[i].toString()));
        }

        Boolean result = Boolean.FALSE;
        Object retObj = null;

        try {
            retObj = dataFilterExpr.evaluate(jc);
        } catch (Throwable e) {
            log.debug("Error occurred when trying to evaluate", dataFilterExpr.toString(), e);
        }

        if(retObj != null && retObj instanceof Boolean) {
            result = (Boolean) retObj;
        }

        return result;
    }

    public Boolean isFilter(Tuple input) throws ExecException {
        if(dataFilterExpr == null) {
            return true;
        }

        if(input == null || input.size() != headers.length) {
            // illegal format data, just skip
            return false;
        }

        jc.clear();

        for(int i = 0; i < input.size(); i++) {
            jc.set(headers[i], ((input.get(i) == null) ? null : input.get(i).toString()));
        }

        Boolean result = Boolean.FALSE;
        Object retObj = null;
        try {
            retObj = dataFilterExpr.evaluate(jc);
        } catch (Throwable e) {
            log.debug("Error occurred when trying to evaluate", dataFilterExpr.toString(), e);
        }

        if(retObj != null && retObj instanceof Boolean) {
            result = (Boolean) retObj;
        }

        return result;
    }

    // reuse context
    public static class ShifuMapContext extends MapContext {
        public ShifuMapContext() {
            super();
        }

        public void clear() {
            if(super.map != null) {
                map.clear();
            }
        }
    }

}