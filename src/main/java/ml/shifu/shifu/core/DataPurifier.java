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

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.util.CommonUtils;

/**
 * DataPurifier class
 */
public class DataPurifier {

    private static Logger log = LoggerFactory.getLogger(DataPurifier.class);

    private static final String NEW_TARGET_TAG_DELIMITER = "|";
    private static final String NEW_TARGET_COLUMN_DELIMITER = "||";
    private static final String NEW_TARGET_DELIMITER = "|||";

    private String[] headers;
    private String dataDelimiter;
    private Expression dataFilterExpr;
    private ShifuMapContext jc = new ShifuMapContext();
    private JexlEngine jexl;

    private final List<ColumnConfig> columnConfigList;

    /**
     * If data purifier is for new tag, not the one set in MC.json.
     */
    private boolean isNewTag = false;

    /**
     * If {@link #isNewTag} is true, what's the new tag column name.
     */
    private String newTagColumnName;

    /**
     * If {@link #isNewTag} is true, new pos tags
     */
    private Set<String> newPosTags;

    /**
     * If {@link #isNewTag} is true, new neg tags
     */
    private Set<String> newNegTags;

    public DataPurifier(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, boolean isForValidationDataSet)
            throws IOException {
        this.columnConfigList = columnConfigList;
        String filterExpressions = (isForValidationDataSet ? modelConfig.getDataSet().getValidationFilterExpressions()
                : modelConfig.getFilterExpressions());
        if(modelConfig.isMultiTask()) {
            filterExpressions = modelConfig.getMTLFilterExpression(modelConfig.getMtlIndex());
        }
        if(StringUtils.isNotBlank(filterExpressions)) {
            filterExpressions = parseNewTagInfo(filterExpressions);
            jexl = new JexlEngine();
            try {
                dataFilterExpr = jexl.createExpression(filterExpressions);
            } catch (JexlException e) {
                log.error("The expression `{}` is invalid, please use correct expression.", filterExpressions, e);
                log.error("Please note the expression won't take effect!");
                dataFilterExpr = null;
            }
            this.headers = CommonUtils.getFinalHeaders(modelConfig);
            dataDelimiter = modelConfig.getDataSetDelimiter();
        }
    }

    private String parseNewTagInfo(String filterExpressions) {
        if(filterExpressions.contains(NEW_TARGET_DELIMITER)) {
            try {
                this.isNewTag = true;
                String[] splits = CommonUtils.split(filterExpressions, NEW_TARGET_DELIMITER);
                filterExpressions = splits[0].trim();

                String[] newTagSplits = CommonUtils.split(splits[1].trim(), NEW_TARGET_COLUMN_DELIMITER);
                this.newTagColumnName = newTagSplits[0].trim();
                if(columnConfigList != null) {
                    ColumnConfig cc = CommonUtils.findColumnConfigByName(columnConfigList, newTagColumnName);
                    if(cc == null) {
                        throw new IllegalArgumentException("'filterExpressions' " + filterExpressions + " with new tag "
                                + this.newTagColumnName + " cannot be found in ColumnConfig.json");
                    }
                }
                String[] newTags = CommonUtils.split(newTagSplits[1].trim(), NEW_TARGET_TAG_DELIMITER);
                this.newPosTags = new HashSet<>();
                for(String tag: newTags) {
                    this.newPosTags.add(tag.trim());
                }
                newTags = CommonUtils.split(newTagSplits[2].trim(), NEW_TARGET_TAG_DELIMITER);
                this.newNegTags = new HashSet<>();
                for(String tag: newTags) {
                    this.newNegTags.add(tag.trim());
                }
            } catch (Exception e) {
                throw new IllegalArgumentException(
                        "Format issue to support new tag segment expassion, the right format should be 'a == 1 |||newTagColumnName||posTag1|posTag2||negTag1|negTag2'.",
                        e);
            }
        }
        return filterExpressions;
    }

    public DataPurifier(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, String filterExpressions,
            boolean strict) throws IOException {
        this.columnConfigList = columnConfigList;
        if(StringUtils.isNotBlank(filterExpressions)) {
            filterExpressions = parseNewTagInfo(filterExpressions);
            jexl = new JexlEngine();
            jexl.setStrict(strict);
            try {
                dataFilterExpr = jexl.createExpression(filterExpressions);
            } catch (JexlException e) {
                if(strict) {
                    throw new RuntimeException(e);
                } else {
                    log.error("The expression " + filterExpressions + "is invalid, please use correct expression.", e);
                }
                dataFilterExpr = null;
            }
            this.headers = CommonUtils.getFinalHeaders(modelConfig);
            dataDelimiter = modelConfig.getDataSetDelimiter();
        }
    }

    public DataPurifier(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, String filterExpressions)
            throws IOException {
        this(modelConfig, columnConfigList, filterExpressions, false);
    }

    public DataPurifier(List<ColumnConfig> columnConfigList, EvalConfig evalConfig) throws IOException {
        this.columnConfigList = columnConfigList;
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getFilterExpressions())) {
            jexl = new JexlEngine();
            try {
                dataFilterExpr = jexl.createExpression(evalConfig.getDataSet().getFilterExpressions());
            } catch (JexlException e) {
                log.error("The expression {} is invalid, please use correct expression.",
                        evalConfig.getDataSet().getFilterExpressions());
                dataFilterExpr = null;
            }

            headers = CommonUtils.getFinalHeaders(evalConfig);
            dataDelimiter = evalConfig.getDataSet().getDataDelimiter();
        }
    }
    
    public DataPurifier(List<ColumnConfig> columnConfigList, EvalConfig evalConfig, int mtlIndex) throws IOException {
        this.columnConfigList = columnConfigList;
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getFilterExpressions())) {
            jexl = new JexlEngine();
            try {
                dataFilterExpr = jexl.createExpression(evalConfig.getDataSet().getFilterExpressions());
            } catch (JexlException e) {
                log.error("The expression {} is invalid, please use correct expression.",
                        evalConfig.getDataSet().getFilterExpressions());
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
            jc.set(headers[i], (fields[i] == null ? "" : fields[i]));
            jc.set(nsColumn.getSimpleName(), (fields[i] == null ? "" : fields[i]));
        }

        Boolean result = Boolean.FALSE;
        Object retObj = null;

        try {
            retObj = dataFilterExpr.evaluate(jc);
        } catch (Throwable e) {
            if(this.jexl.isStrict()) {
                throw new RuntimeException(e);
            } else {
                log.error("Error occurred when trying to evaluate " + dataFilterExpr.toString(), e);
            }
        }

        if(retObj != null && retObj instanceof Boolean) {
            result = (Boolean) retObj;
        } else if(retObj != null && !(retObj instanceof Boolean)) {
            throw new InvalidFilterResultExcetion(
                    "Invalid filter return not boolean type: " + dataFilterExpr.getExpression());
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
            if(this.jexl.isStrict()) {
                throw new RuntimeException(e);
            } else {
                log.warn("Error occurred when trying to evaluate " + dataFilterExpr.toString(), e);
            }
        }

        if(retObj != null && retObj instanceof Boolean) {
            result = (Boolean) retObj;
        } else if(retObj != null && !(retObj instanceof Boolean)) {
            throw new InvalidFilterResultExcetion(
                    "Invalid filter return not boolean type: " + dataFilterExpr.getExpression());
        }

        return result;
    }

    // reuse context
    public static class InvalidFilterResultExcetion extends RuntimeException {

        private static final long serialVersionUID = 279485512893373010L;

        /**
         * Constructs a new fail task runtime exception with <code>null</code> as its detail message. The cause is
         * not initialized, and may subsequently be initialized by a call to {@link #initCause}.
         */
        public InvalidFilterResultExcetion() {
        }

        /**
         * Constructs a new fail task runtime exception with the specified detail message. The cause is not
         * initialized, and may subsequently be initialized by a call to {@link #initCause}.
         * 
         * @param message
         *            the detail message. The detail message is saved for later retrieval by the {@link #getMessage()}
         *            method.
         */
        public InvalidFilterResultExcetion(String message) {
            super(message);
        }

        /**
         * Constructs a new fail task runtime exception with the specified detail message and cause.
         * <p>
         * Note that the detail message associated with <code>cause</code> is <i>not</i> automatically incorporated in
         * this runtime exception's detail message.
         * 
         * @param message
         *            the detail message (which is saved for later retrieval by the {@link #getMessage()} method).
         * @param cause
         *            the cause (which is saved for later retrieval by the {@link #getCause()} method). (A <tt>null</tt>
         *            value is permitted, and indicates that the cause is nonexistent or unknown.)
         */
        public InvalidFilterResultExcetion(String message, Throwable cause) {
            super(message, cause);
        }

        /**
         * Constructs a new fail task runtime exception with the specified cause and a detail message of
         * <tt>(cause==null ? null : cause.toString())</tt> (which typically contains the class and detail
         * message of <tt>cause</tt>). This constructor is useful for runtime exceptions that are little more than
         * wrappers for other throwables.
         * 
         * @param cause
         *            the cause (which is saved for later retrieval by the {@link #getCause()} method). (A <tt>null</tt>
         *            value is permitted, and indicates that the cause is nonexistent or unknown.)
         */
        public InvalidFilterResultExcetion(Throwable cause) {
            super(cause);
        }

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

    /**
     * @return the isNewTag
     */
    public boolean isNewTag() {
        return isNewTag;
    }

    /**
     * @return the newTagColumnName
     */
    public String getNewTagColumnName() {
        return newTagColumnName;
    }

    /**
     * @return the newPosTags
     */
    public Set<String> getNewPosTags() {
        return newPosTags;
    }

    /**
     * @return the newNegTags
     */
    public Set<String> getNewNegTags() {
        return newNegTags;
    }

}