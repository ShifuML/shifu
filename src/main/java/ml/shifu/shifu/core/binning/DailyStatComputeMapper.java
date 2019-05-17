/*
 * Copyright [2012-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.binning;

import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.autotype.AutoTypeDistinctCountMapper.CountAndFrequentItems;
import ml.shifu.shifu.core.autotype.CountAndFrequentItemsWritable;
import ml.shifu.shifu.util.BinUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.MapReduceUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

/**
 * {@link DailyStatComputeMapper} is a mapper to update local data statistics given bin boundary list.
 * 
 * <p>
 * Bin boundary list is got by using distributed cache. After read bin boundary list, by iterate each record, to update
 * count and weighted value in each bin.
 * 
 * <p>
 * This map-reduce job is to solve issue in group by all data together per each column in pig version. It is job with
 * mappers and one reducer. The scalability is very good.
 * 
 * <p>
 * We assume that all column info can be saved in mapper memory.
 * 
 * <p>
 * 'median' can not be computed through such distributed solution.
 */
public class DailyStatComputeMapper extends Mapper<LongWritable, Text, Text, DailyStatInfoWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(DailyStatComputeMapper.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    private String dataSetDelimiter;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * To filter records by customized expressions
     */
    private DataPurifier dataPurifier;

    /**
     * Weight column index.
     */
    private int weightedColumnNum = -1;

    /**
     * Date column index.
     */
    private int dateColumnNum = -1;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Tag column index
     */
    private int tagColumnNum = -1;

    /**
     * A map to store all statistics for all columns.
     */
    private Map<String, DailyStatInfoWritable> dailyStatInfo;

    /**
     * Output key cache to avoid new operation.
     */
    private Text outputKey;

    // cache tags in set for search
    private Set<String> posTags;
    private Set<String> negTags;
    private Set<String> tags;
    private Set<String> missingOrInvalidValues;

    private int weightExceptions = 0;
    private boolean isThrowforWeightException;
    private boolean isLinearTarget = false;

    /**
     * Data purifiers for column expansion
     */
    private List<DataPurifier> expressionDataPurifiers;
    private boolean isForExpressions = false;

    /**
     * Load model config and column config files.
     */
    private void loadConfigFiles(final Context context) {
        try {
            SourceType sourceType = SourceType.valueOf(
                    context.getConfiguration().get(Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(context.getConfiguration().get(Constants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Initialization for column statistics in mapper.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.dataSetDelimiter = this.modelConfig.getDataSetDelimiter();

        this.dataPurifier = new DataPurifier(this.modelConfig, false);

        String filterExpressions = context.getConfiguration().get(Constants.SHIFU_STATS_FILTER_EXPRESSIONS);
        if(StringUtils.isNotBlank(filterExpressions)) {
            this.isForExpressions = true;
            String[] splits = CommonUtils.split(filterExpressions, Constants.SHIFU_STATS_FILTER_EXPRESSIONS_DELIMETER);
            this.expressionDataPurifiers = new ArrayList<DataPurifier>(splits.length);
            for(String split: splits) {
                this.expressionDataPurifiers.add(new DataPurifier(modelConfig, split, false));
            }
        }

        loadWeightColumnNum();

        loadTagWeightNum();

        loadDateColumnNum();

        this.dailyStatInfo = new HashMap<String, DailyStatInfoWritable>(this.columnConfigList.size(), 1f);


        this.outputKey = new Text();

        this.posTags = new HashSet<String>(modelConfig.getPosTags());
        this.negTags = new HashSet<String>(modelConfig.getNegTags());
        this.tags = new HashSet<String>(modelConfig.getFlattenTags());

        this.missingOrInvalidValues = new HashSet<String>(this.modelConfig.getDataSet().getMissingOrInvalidValues());

        this.isThrowforWeightException = "true"
                .equalsIgnoreCase(context.getConfiguration().get("shifu.weight.exception", "false"));

        LOG.debug("Daily stat info: {}", this.dailyStatInfo);
        this.isLinearTarget = (CollectionUtils.isEmpty(modelConfig.getTags())
                && CommonUtils.getTargetColumnConfig(columnConfigList).isNumerical());
    }

    /**
     * Load tag weight index field.
     */
    private void loadTagWeightNum() {
        for(ColumnConfig config: this.columnConfigList) {
            if(config.isTarget()) {
                this.tagColumnNum = config.getColumnNum();
                break;
            }
        }

        if(this.tagColumnNum == -1) {
            throw new RuntimeException("No valid target column.");
        }
    }

    /**
     * Load weight column index field.
     */
    private void loadWeightColumnNum() {
        String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();
        if(weightColumnName != null && weightColumnName.length() != 0) {
            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                if(config.getColumnName().equals(weightColumnName)) {
                    this.weightedColumnNum = i;
                    break;
                }
            }
        }
    }

    /**
     * Load date column index field.
     */
    private void loadDateColumnNum() {
        String dateColumnName = this.modelConfig.getDataSet().getDateColumnName();
        if(dateColumnName != null && dateColumnName.length() != 0) {
            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                if(config.getColumnName().equals(dateColumnName)) {
                    this.dateColumnNum = i;
                    break;
                }
            }
        }
    }

    /**
     * Mapper implementation includes: 1. Invalid data purifier 2. Column statistics update.
     */
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        LOG.info("Start map." + key + " " + value);
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }

        context.getCounter(Constants.SHIFU_GROUP_COUNTER, "TOTAL_VALID_COUNT").increment(1L);

        if(!this.dataPurifier.isFilter(valueStr)) {
            context.getCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT").increment(1L);
            return;
        }

        String[] units = CommonUtils.split(valueStr, this.dataSetDelimiter);
        // tagColumnNum should be in units array, if not IndexOutofBoundException
        if(units.length != this.columnConfigList.size()) {
            LOG.error("Data column length doesn't match with ColumnConfig size. Just skip.");
            return;
        }

        String tag = CommonUtils.trimTag(units[this.tagColumnNum]);

        if(modelConfig.isRegression()) {
            if(tag == null || (!posTags.contains(tag) && !negTags.contains(tag))) {
                context.getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1L);
                return;
            }
        } else {
            if(tag == null || (!isLinearTarget && !tags.contains(tag))) {
                context.getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1L);
                return;
            }
        }

        Double weight = 1.0;
        try {
            weight = (this.weightedColumnNum == -1 ? 1.0d : Double.valueOf(units[this.weightedColumnNum]));
            if(weight < 0) {
                weightExceptions += 1;
                context.getCounter(Constants.SHIFU_GROUP_COUNTER, "WEIGHT_EXCEPTION").increment(1L);
                if(weightExceptions > 5000 && this.isThrowforWeightException) {
                    throw new IllegalStateException(
                            "Please check weight column in eval, exceptional weight count is over 5000");
                }
            }
        } catch (NumberFormatException e) {
            weightExceptions += 1;
            context.getCounter(Constants.SHIFU_GROUP_COUNTER, "WEIGHT_EXCEPTION").increment(1L);
            if(weightExceptions > 5000 && this.isThrowforWeightException) {
                throw new IllegalStateException(
                        "Please check weight column in eval, exceptional weight count is over 5000");
            }
        }

        List<Boolean> filterResults = null;
        if(this.isForExpressions) {
            filterResults = new ArrayList<Boolean>();
            for(int j = 0; j < this.expressionDataPurifiers.size(); j++) {
                DataPurifier dp = this.expressionDataPurifiers.get(j);
                filterResults.add(dp.isFilter(valueStr));
            }
        }


        // valid data process
        for(int i = 0; i < units.length; i++) {
            populateStats(units, tag, weight, i, i);
            if(this.isForExpressions) {
                for(int j = 0; j < this.expressionDataPurifiers.size(); j++) {
                    Boolean filter = filterResults.get(j);
                    if(filter != null && filter) {
                        populateStats(units, tag, weight, i, (j + 1) * units.length + i);
                    }
                }
            }
        }
    }

    private void populateStats(String[] units, String tag, Double weight, int columnIndex, int newCCIndex) {
        ColumnConfig columnConfig = this.columnConfigList.get(columnIndex);

        boolean isMissingValue = false;
        boolean isInvalidValue = false;

        String variableName = columnConfig.getColumnName().toLowerCase();
        DailyStatInfoWritable dailyStatInfoWritable = this.dailyStatInfo.get(variableName);

        if(dailyStatInfoWritable == null) {
            dailyStatInfoWritable = new DailyStatInfoWritable();
            this.dailyStatInfo.put(variableName, dailyStatInfoWritable);
        }
        String dateVal = "";
        if(dateColumnNum >= 0) {
            dateVal = units[dateColumnNum].toLowerCase();
        }
        Map<String, DailyStatInfoWritable.VariableStatInfo> map = dailyStatInfoWritable.getVariableDailyStatInfo();
        DailyStatInfoWritable.VariableStatInfo variableStatInfo = map.get(dateVal);
        if(variableStatInfo == null){
            variableStatInfo = new DailyStatInfoWritable.VariableStatInfo();
            map.put(dateVal, variableStatInfo);
        }
        LOG.info("columnIndex="+columnIndex);
        variableStatInfo.setColumnConfigIndex(columnIndex);
        variableStatInfo.setTotalCount(variableStatInfo.getTotalCount() + 1L);

        if(columnConfig.isCategorical()) {
            int lastBinIndex = columnConfig.getBinCategory().size();
            variableStatInfo.init(lastBinIndex);
            int binNum = 0;
            if(units[columnIndex] == null || missingOrInvalidValues.contains(variableName)) {
                isMissingValue = true;
            } else {
                String str = units[columnIndex];
                binNum = quickLocateCategoricalBin(columnConfig.getBinCategory(), str);
                if(binNum < 0) {
                    isInvalidValue = true;
                }
            }

            if(isInvalidValue || isMissingValue) {
                variableStatInfo.setMissingCount(variableStatInfo.getMissingCount() + 1L);
                binNum = lastBinIndex;
            }

            LOG.info("isRegression:" + modelConfig.isRegression() + ", tag:" + tag + ", posTags:" + posTags.size() + ", negTags:" + negTags.size() + ", binNum:" + binNum);
            if(modelConfig.isRegression()) {
                if(posTags.contains(tag)) {
                    variableStatInfo.getBinCountPos()[binNum] += 1L;
                    variableStatInfo.getBinWeightPos()[binNum] += weight;
                } else if(negTags.contains(tag)) {
                    variableStatInfo.getBinCountNeg()[binNum] += 1L;
                    variableStatInfo.getBinWeightNeg()[binNum] += weight;
                }
            } else {
                // for multiple classification, set bin count to BinCountPos and leave BinCountNeg empty
                variableStatInfo.getBinCountPos()[binNum] += 1L;
                variableStatInfo.getBinWeightPos()[binNum] += weight;
            }
        } else if(columnConfig.isNumerical()) {
            int lastBinIndex = columnConfig.getBinBoundary().size();
            variableStatInfo.init(lastBinIndex);
            double douVal = 0.0;
            if(units[columnIndex] == null || units[columnIndex].length() == 0
                    || missingOrInvalidValues.contains(units[columnIndex].toLowerCase())) {
                isMissingValue = true;
            } else {
                try {
                    douVal = Double.parseDouble(units[columnIndex].trim());
                } catch (Exception e) {
                    isInvalidValue = true;
                }
            }

            // add logic the same as CalculateNewStatsUDF
            if(Double.compare(douVal, modelConfig.getNumericalValueThreshold()) > 0) {
                isInvalidValue = true;
            }

            if(isInvalidValue || isMissingValue) {
                variableStatInfo.setMissingCount(variableStatInfo.getMissingCount() + 1L);
                if(modelConfig.isRegression()) {
                    if(posTags.contains(tag)) {
                        variableStatInfo.getBinCountPos()[lastBinIndex] += 1L;
                        variableStatInfo.getBinWeightPos()[lastBinIndex] += weight;
                    } else if(negTags.contains(tag)) {
                        variableStatInfo.getBinCountNeg()[lastBinIndex] += 1L;
                        variableStatInfo.getBinWeightNeg()[lastBinIndex] += weight;
                    }
                }
            } else {
                // For invalid or missing values, no need update sum, squaredSum, max, min ...
                int binNum = getBinNum(columnConfig.getBinBoundary(), units[columnIndex]);
                if(binNum == -1) {
                    throw new RuntimeException("binNum should not be -1 to this step.");
                }
                if(modelConfig.isRegression()) {
                    if(posTags.contains(tag)) {
                        variableStatInfo.getBinCountPos()[binNum] += 1L;
                        variableStatInfo.getBinWeightPos()[binNum] += weight;
                    } else if(negTags.contains(tag)) {
                        variableStatInfo.getBinCountNeg()[binNum] += 1L;
                        variableStatInfo.getBinWeightNeg()[binNum] += weight;
                    }
                }
                variableStatInfo.setSum(variableStatInfo.getSum() + douVal);
                double squaredVal = douVal * douVal;
                variableStatInfo.setSquaredSum(variableStatInfo.getSquaredSum() + squaredVal);
                variableStatInfo.setTripleSum(variableStatInfo.getTripleSum() + squaredVal * douVal);
                variableStatInfo.setQuarticSum(variableStatInfo.getQuarticSum() + squaredVal * squaredVal);

                if(Double.compare(variableStatInfo.getMax(), douVal) < 0) {
                    variableStatInfo.setMax(douVal);
                }
                if(Double.compare(variableStatInfo.getMin(), douVal) > 0) {
                    variableStatInfo.setMin(douVal);
                }
            }
        }
    }

    public static int getBinNum(List<Double> binBoundaryList, String columnVal) {
        if(StringUtils.isBlank(columnVal)) {
            return -1;
        }
        double dval = 0.0;
        try {
            dval = Double.parseDouble(columnVal);
        } catch (Exception e) {
            return -1;
        }
        return BinUtils.getBinIndex(binBoundaryList, dval);
    }

    public static int getBinNum(List<Double> binBoundaryList, double dVal) {
        return BinUtils.getBinIndex(binBoundaryList, dVal);
    }

    private int quickLocateCategoricalBin(List<String> list, String val) {
        for (int i = 0; i < list.size(); i++){
            if(StringUtils.equals(list.get(i), val)){
                return i;
            }
        }
        return -1;
    }

    /**
     * Write column info to reducer for merging.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        LOG.debug("Column binning info: {}", this.dailyStatInfo);

        for(Map.Entry<String, DailyStatInfoWritable> entry: this.dailyStatInfo.entrySet()) {
            this.outputKey.set(entry.getKey());
            for(Map.Entry<String, DailyStatInfoWritable.VariableStatInfo> inEntry : entry.getValue().getVariableDailyStatInfo().entrySet()){

                LOG.info("output." + entry.getKey() + " " + inEntry.getKey() + " " + inEntry.getValue());
            }

            context.write(this.outputKey, entry.getValue());
        }
    }
}
