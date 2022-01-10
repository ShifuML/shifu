/*
 * Copyright [2012-2014] PayPal Software Foundation
 * <p/>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.tools.pigstats.PigStatusReporter;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.udf.norm.PrecisionType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

/**
 * <pre>
 * AddColumnNumUDF class is to convert tuple of row data into bag of column data
 * Its structure is like
 *    {
 *         (column-id, column-value, column-tag, column-score)
 *         (column-id, column-value, column-tag, column-score)
 *         ...
 * }
 * </pre>
 */
public class AddColumnNumAndFilterUDF extends AddColumnNumUDF {

    private static final int MAX_MISMATCH_CNT = 500;

    private final boolean isAppendRandom;

    private List<DataPurifier> dataPurifiers;

    private List<Integer> newTagIndexes;

    private boolean isForExpressions = false;
    private boolean isLinearTarget = false;
    private int mismatchCnt = 0;

    // null means not set
    private PrecisionType precisionType;

    public AddColumnNumAndFilterUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr)
            throws Exception {
        this(source, pathModelConfig, pathColumnConfig, withScoreStr, "true");
    }

    public AddColumnNumAndFilterUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr,
            String isAppendRandom) throws Exception {
        super(source, pathModelConfig, pathColumnConfig, withScoreStr);
        this.isAppendRandom = Boolean.TRUE.toString().equalsIgnoreCase(isAppendRandom);

        String filterExpressions;
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            filterExpressions = UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_SEGMENT_EXPRESSIONS);
        } else {
            filterExpressions = Environment.getProperty(Constants.SHIFU_SEGMENT_EXPRESSIONS);
        }

        String precision = getUdfProperty(Constants.SHIFU_PRECISION_TYPE);
        if(StringUtils.isNotBlank(precision)) {
            this.precisionType = PrecisionType
                    .of(getUdfProperty(Constants.SHIFU_PRECISION_TYPE, PrecisionType.FLOAT32.toString()));
        }

        if(StringUtils.isNotBlank(filterExpressions)) {
            this.isForExpressions = true;
            String[] splits = CommonUtils.split(filterExpressions, Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
            this.dataPurifiers = new ArrayList<>(splits.length);
            this.newTagIndexes = new ArrayList<>(splits.length);
            for(String split: splits) {
                DataPurifier dataPurifier = new DataPurifier(modelConfig, this.columnConfigList, split, false);
                if(dataPurifier.isNewTag()) {
                    ColumnConfig cc = CommonUtils.findColumnConfigByName(columnConfigList,
                            dataPurifier.getNewTagColumnName());
                    this.newTagIndexes.add(cc == null ? -1 : cc.getColumnNum());
                } else {
                    this.newTagIndexes.add(-1);
                }
                this.dataPurifiers.add(dataPurifier);
            }
        }

        this.isLinearTarget = CommonUtils.isLinearTarget(modelConfig, columnConfigList);
    }

    @SuppressWarnings("deprecation")
    @Override
    public DataBag exec(Tuple input) throws IOException {
        DataBag bag = BagFactory.getInstance().newDefaultBag();
        TupleFactory tupleFactory = TupleFactory.getInstance();

        if(input == null) {
            return null;
        }

        int size = input.size();

        if(size == 0 || input.size() != this.columnConfigList.size()) {
            log.error("the input size - " + input.size() + ", while column size - " + columnConfigList.size());
            this.mismatchCnt++;

            // Throw exceptions if the mismatch count is greater than MAX_MISMATCH_CNT,
            // this could make Shifu could skip some malformed data
            if(this.mismatchCnt > MAX_MISMATCH_CNT) {
                throw new ShifuException(ShifuErrorCode.ERROR_NO_EQUAL_COLCONFIG);
            }
            return null;
        }

        if(input.get(tagColumnNum) == null) {
            log.error("tagColumnNum is " + tagColumnNum + "; input size is " + input.size()
                    + "; columnConfigList.size() is " + columnConfigList.size() + "; tuple is"
                    + (System.currentTimeMillis() % 100 == 0 ? "" : input.toDelimitedString("|")) + "; tag is "
                    + input.get(tagColumnNum));
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        String tag = CommonUtils.trimTag(input.get(tagColumnNum).toString());
        if(this.isLinearTarget) {
            if(!NumberUtils.isNumber(tag)) {
                if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                    PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")
                            .increment(1);
                }
                return null;
            }
        } else if(!super.tagSet.contains(tag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1);
            }
            return null;
        }

        Double rate = modelConfig.getBinningSampleRate();
        if(!this.isLinearTarget && !modelConfig.isClassification() && modelConfig.isBinningSampleNegOnly()) {
            if(super.negTagSet.contains(tag) && random.nextDouble() > rate) {
                return null;
            }
        } else {
            if(random.nextDouble() > rate) {
                return null;
            }
        }

        List<Boolean> filterResultList = null;
        if(this.isForExpressions) {
            filterResultList = new ArrayList<Boolean>();
            for(int j = 0; j < this.dataPurifiers.size(); j++) {
                DataPurifier dataPurifier = this.dataPurifiers.get(j);
                filterResultList.add(dataPurifier.isFilter(input));
            }
        }

        boolean isPositiveInst = (modelConfig.isRegression() && super.posTagSet.contains(tag));
        for(int i = 0; i < size; i++) {
            ColumnConfig config = columnConfigList.get(i);
            if(!isValidRecord(modelConfig.isRegression(), isPositiveInst, config)) {
                continue;
            }

            bag.add(buildTuple(input, tupleFactory, this.posTagSet, this.negTagSet, tag, i, i));
            if(this.isForExpressions) {
                for(int j = 0; j < this.dataPurifiers.size(); j++) {
                    DataPurifier dataPurifier = this.dataPurifiers.get(j);
                    Boolean isFilter = filterResultList.get(j);
                    if(isFilter != null && isFilter) {
                        if(dataPurifier.isNewTag()) {
                            String newTag = CommonUtils.trimTag(input.get(this.newTagIndexes.get(j)).toString());
                            Set<String> newPosTags = dataPurifier.getNewPosTags();
                            Set<String> newNegTags = dataPurifier.getNewNegTags();
                            if(newTag == null || (!newPosTags.contains(newTag) && !newNegTags.contains(newTag))) {
                                if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_EXTENSION_TAG")) {
                                    PigStatusReporter.getInstance()
                                            .getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_EXTENSION_TAG")
                                            .increment(1);
                                }
                            } else {
                                bag.add(buildTuple(input, tupleFactory, newPosTags, newNegTags, newTag, i,
                                        (j + 1) * size + i));
                            }
                        } else {
                            bag.add(buildTuple(input, tupleFactory, this.posTagSet, this.negTagSet, tag, i,
                                    (j + 1) * size + i));
                        }
                    }
                }
            }
        }
        return bag;
    }

    private Tuple buildTuple(Tuple input, TupleFactory tupleFactory, Set<String> posTags, Set<String> negTags,
            String tag, int i, int finalIndex) throws ExecException {
        Tuple tuple = tupleFactory.newTuple(TOTAL_COLUMN_CNT);
        tuple.set(COLUMN_ID_INDX, finalIndex);

        if(this.precisionType == null || this.columnConfigList.get(i).isCategorical()) {
            tuple.set(COLUMN_VAL_INDX, (input.get(i) == null ? null : input.get(i).toString())); // Set Data
        } else {
            double dval;
            try {
                dval = Double.parseDouble(input.get(i).toString());
                // reduced to precision Type
                tuple.set(COLUMN_VAL_INDX, (input.get(i) == null ? null : this.precisionType.to(dval).toString())); 
            } catch (Exception e) {
                tuple.set(COLUMN_VAL_INDX, null); // Set Data
            }
        }

        if(modelConfig.isRegression()) {
            if(posTags.contains(tag)) { // Set Tag
                tuple.set(COLUMN_TAG_INDX, true);
            }

            if(negTags.contains(tag)) {
                tuple.set(COLUMN_TAG_INDX, false);
            }
        } else {
            // a mock for multiple classification and linear target
            tuple.set(COLUMN_TAG_INDX, true);
        }
        tuple.set(COLUMN_WEIGHT_INDX, getWeightColumnVal(input)); // get weight value

        // add random seed for distribution for bigger mapper, 300 is not enough TODO
        if(this.isAppendRandom) {
            tuple.set(COLUMN_SEED_INDX, Math.abs(random.nextInt() % 300));
        }
        return tuple;
    }

    @Override
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new FieldSchema("value", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema("tag", DataType.BOOLEAN));
            if(this.isAppendRandom) {
                tupleSchema.add(new FieldSchema("rand", DataType.INTEGER));
            }
            tupleSchema.add(new FieldSchema("weight", DataType.DOUBLE));
            return new Schema(new Schema.FieldSchema("columnInfos",
                    new Schema(new Schema.FieldSchema("columnInfo", tupleSchema, DataType.TUPLE)), DataType.BAG));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

    private boolean isValidRecord(boolean isBinary, boolean isPositive, ColumnConfig columnConfig) {
        if(isBinary) {
            return columnConfig != null && (columnConfig.isCategorical() || isValidBinningMethodForBinary(isPositive));
        } else {
            return columnConfig != null && (columnConfig.isCategorical() || isValidBinningMethod());
        }
    }

    private boolean isValidBinningMethodForBinary(boolean isPositive) {
        return modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.DynamicBinning)
                || modelConfig.getBinningMethod().equals(BinningMethod.EqualTotal)
                || modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)
                || (modelConfig.getBinningMethod().equals(BinningMethod.EqualPositive) && isPositive)
                || (modelConfig.getBinningMethod().equals(BinningMethod.EqualNegative) && !isPositive)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualTotal)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualInterval)
                || (modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualPositive) && isPositive)
                || (modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualNegative) && !isPositive);
    }

    private boolean isValidBinningMethod() {
        return modelConfig.getBinningMethod().equals(BinningMethod.EqualTotal)
                || modelConfig.getBinningMethod().equals(BinningMethod.EqualInterval)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualTotal)
                || modelConfig.getBinningMethod().equals(BinningMethod.WeightEqualInterval);
    }
}
