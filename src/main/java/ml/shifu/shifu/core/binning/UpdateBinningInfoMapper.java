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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.autotype.AutoTypeDistinctCountMapper.CountAndFrequentItems;
import ml.shifu.shifu.core.autotype.CountAndFrequentItemsWritable;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link UpdateBinningInfoMapper} is a mapper to update local data statistics given bin boundary list.
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
public class UpdateBinningInfoMapper extends Mapper<LongWritable, Text, IntWritable, BinningInfoWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(UpdateBinningInfoMapper.class);

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
    private Map<Integer, BinningInfoWritable> columnBinningInfo;

    /**
     * Bin boundary list splitter.
     */
    private static Splitter BIN_BOUNDARY_SPLITTER = Splitter.on(Constants.BIN_BOUNDRY_DELIMITER).trimResults();

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /*
     * TODO At risk to be a big memory cost OOM.
     */
    private Map<Integer, Map<String, Integer>> categoricalBinMap;

    /**
     * Using approximate method to estimate real frequent items and store into this map
     */
    private Map<Integer, CountAndFrequentItems> variableCountMap;

    // cache tags in set for search
    private Set<String> posTags;
    private Set<String> negTags;
    private Set<String> tags;
    private Set<String> missingOrInvalidValues;

    private int weightExceptions = 0;
    private boolean isThrowforWeightException;

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
            SourceType sourceType = SourceType.valueOf(context.getConfiguration().get(
                    Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(
                    context.getConfiguration().get(Constants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);
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

        this.dataPurifier = new DataPurifier(this.modelConfig);

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

        this.columnBinningInfo = new HashMap<Integer, BinningInfoWritable>(this.columnConfigList.size(), 1f);
        this.categoricalBinMap = new HashMap<Integer, Map<String, Integer>>(this.columnConfigList.size(), 1f);

        loadColumnBinningInfo();

        this.outputKey = new IntWritable();

        this.variableCountMap = new HashMap<Integer, CountAndFrequentItems>();

        this.posTags = new HashSet<String>(modelConfig.getPosTags());
        this.negTags = new HashSet<String>(modelConfig.getNegTags());
        this.tags = new HashSet<String>(modelConfig.getFlattenTags());

        this.missingOrInvalidValues = new HashSet<String>(this.modelConfig.getDataSet().getMissingOrInvalidValues());

        this.isThrowforWeightException = "true".equalsIgnoreCase(context.getConfiguration().get(
                "shifu.weight.exception", "false"));

        LOG.debug("Column binning info: {}", this.columnBinningInfo);
    }

    /**
     * Load and initialize column binning info object.
     */
    private void loadColumnBinningInfo() throws FileNotFoundException, IOException {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(Constants.BINNING_INFO_FILE_NAME),
                    Charset.forName("UTF-8")));
            String line = reader.readLine();
            while(line != null && line.length() != 0) {
                LOG.debug("line is {}", line);
                // here just use String.split for just two columns
                String[] cols = CommonUtils.split(line.trim(), Constants.DEFAULT_DELIMITER);
                if(cols != null && cols.length >= 2) {
                    Integer rawColumnNum = Integer.parseInt(cols[0]);
                    BinningInfoWritable binningInfo = new BinningInfoWritable();
                    int corrColumnNum = rawColumnNum;
                    if(rawColumnNum >= this.columnConfigList.size()) {
                        corrColumnNum = rawColumnNum % this.columnConfigList.size();
                    }
                    binningInfo.setColumnNum(rawColumnNum);
                    ColumnConfig columnConfig = this.columnConfigList.get(corrColumnNum);
                    int binSize = 0;
                    if(columnConfig.isHybrid()) {
                        binningInfo.setNumeric(true);
                        String[] splits = CommonUtils.split(cols[1], Constants.HYBRID_BIN_STR_DILIMETER);

                        List<Double> list = new ArrayList<Double>();
                        for(String startElement: BIN_BOUNDARY_SPLITTER.split(splits[0])) {
                            list.add(Double.valueOf(startElement));
                        }
                        binningInfo.setBinBoundaries(list);

                        List<String> cateList = new ArrayList<String>();
                        Map<String, Integer> map = this.categoricalBinMap.get(rawColumnNum);
                        if(map == null) {
                            map = new HashMap<String, Integer>();
                            this.categoricalBinMap.put(rawColumnNum, map);
                        }
                        int index = 0;
                        if(!StringUtils.isBlank(splits[1])) {
                            for(String startElement: BIN_BOUNDARY_SPLITTER.split(splits[1])) {
                                cateList.add(startElement);
                                map.put(startElement, index++);
                            }
                        }
                        binningInfo.setBinCategories(cateList);
                        binSize = list.size() + cateList.size();
                    } else if(columnConfig.isNumerical()) {
                        binningInfo.setNumeric(true);
                        List<Double> list = new ArrayList<Double>();
                        for(String startElement: BIN_BOUNDARY_SPLITTER.split(cols[1])) {
                            list.add(Double.valueOf(startElement));
                        }
                        binningInfo.setBinBoundaries(list);
                        binSize = list.size();
                    } else {
                        binningInfo.setNumeric(false);
                        List<String> list = new ArrayList<String>();
                        Map<String, Integer> map = this.categoricalBinMap.get(rawColumnNum);
                        if(map == null) {
                            map = new HashMap<String, Integer>();
                            this.categoricalBinMap.put(rawColumnNum, map);
                        }
                        int index = 0;
                        if(!StringUtils.isBlank(cols[1])) {
                            for(String startElement: BIN_BOUNDARY_SPLITTER.split(cols[1])) {
                                list.add(startElement);
                                map.put(startElement, index++);
                            }
                        }
                        binningInfo.setBinCategories(list);
                        binSize = list.size();
                    }
                    long[] binCountPos = new long[binSize + 1];
                    binningInfo.setBinCountPos(binCountPos);

                    long[] binCountNeg = new long[binSize + 1];
                    binningInfo.setBinCountNeg(binCountNeg);

                    double[] binWeightPos = new double[binSize + 1];
                    binningInfo.setBinWeightPos(binWeightPos);

                    double[] binWeightNeg = new double[binSize + 1];
                    binningInfo.setBinWeightNeg(binWeightNeg);
                    LOG.debug("column num {}  and info {}", rawColumnNum, binningInfo);
                    this.columnBinningInfo.put(rawColumnNum, binningInfo);
                }
                line = reader.readLine();
            }
        } finally {
            if(reader != null) {
                reader.close();
            }
        }
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
     * Mapper implementation includes: 1. Invalid data purifier 2. Column statistics update.
     */
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
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
        if ( units.length != this.columnConfigList.size() ) {
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
            if(tag == null || (!tags.contains(tag))) {
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

        CountAndFrequentItems countAndFrequentItems = this.variableCountMap.get(newCCIndex);
        if(countAndFrequentItems == null) {
            countAndFrequentItems = new CountAndFrequentItems();
            this.variableCountMap.put(newCCIndex, countAndFrequentItems);
        }
        countAndFrequentItems.offer(this.missingOrInvalidValues, units[columnIndex]);

        boolean isMissingValue = false;
        boolean isInvalidValue = false;

        BinningInfoWritable binningInfoWritable = this.columnBinningInfo.get(newCCIndex);
        if(binningInfoWritable == null) {
            return;
        }
        binningInfoWritable.setTotalCount(binningInfoWritable.getTotalCount() + 1L);
        if(columnConfig.isHybrid()) {
            int binNum = 0;
            if(units[columnIndex] == null || missingOrInvalidValues.contains(units[columnIndex].toLowerCase())) {
                isMissingValue = true;
            }
            String str = StringUtils.trim(units[columnIndex]);
            double douVal = CommonUtils.parseNumber(str);

            Double hybridThreshould = columnConfig.getHybridThreshold();
            if(hybridThreshould == null) {
                hybridThreshould = Double.NEGATIVE_INFINITY;
            }
            // douVal < hybridThreshould which will also be set to category
            boolean isCategory = Double.isNaN(douVal) || douVal < hybridThreshould;
            boolean isNumber = !Double.isNaN(douVal);

            if(isMissingValue) {
                binningInfoWritable.setMissingCount(binningInfoWritable.getMissingCount() + 1L);
                binNum = binningInfoWritable.getBinCategories().size() + binningInfoWritable.getBinBoundaries().size();
            } else if(isCategory) {
                // get categorical bin number in category list
                binNum = quickLocateCategoricalBin(this.categoricalBinMap.get(newCCIndex), str);
                if(binNum < 0) {
                    isInvalidValue = true;
                }
                if(isInvalidValue) {
                    // the same as missing count
                    binningInfoWritable.setMissingCount(binningInfoWritable.getMissingCount() + 1L);
                    binNum = binningInfoWritable.getBinCategories().size()
                            + binningInfoWritable.getBinBoundaries().size();
                } else {
                    // if real category value, binNum should + binBoundaries.size
                    binNum += binningInfoWritable.getBinBoundaries().size();;
                }
            } else if(isNumber) {
                binNum = getBinNum(binningInfoWritable.getBinBoundaries(), douVal);
                if(binNum == -1) {
                    throw new RuntimeException("binNum should not be -1 to this step.");
                }

                // other stats are treated as numerical features
                binningInfoWritable.setSum(binningInfoWritable.getSum() + douVal);
                double squaredVal = douVal * douVal;
                binningInfoWritable.setSquaredSum(binningInfoWritable.getSquaredSum() + squaredVal);
                binningInfoWritable.setTripleSum(binningInfoWritable.getTripleSum() + squaredVal * douVal);
                binningInfoWritable.setQuarticSum(binningInfoWritable.getQuarticSum() + squaredVal * squaredVal);

                if(Double.compare(binningInfoWritable.getMax(), douVal) < 0) {
                    binningInfoWritable.setMax(douVal);
                }
                if(Double.compare(binningInfoWritable.getMin(), douVal) > 0) {
                    binningInfoWritable.setMin(douVal);
                }
            }
            if(posTags.contains(tag)) {
                binningInfoWritable.getBinCountPos()[binNum] += 1L;
                binningInfoWritable.getBinWeightPos()[binNum] += weight;
            } else if(negTags.contains(tag)) {
                binningInfoWritable.getBinCountNeg()[binNum] += 1L;
                binningInfoWritable.getBinWeightNeg()[binNum] += weight;
            }
        } else if(columnConfig.isCategorical()) {
            int lastBinIndex = binningInfoWritable.getBinCategories().size();

            int binNum = 0;
            if(units[columnIndex] == null || missingOrInvalidValues.contains(units[columnIndex].toLowerCase())) {
                isMissingValue = true;
            } else {
                String str = StringUtils.trim(units[columnIndex]);
                binNum = quickLocateCategoricalBin(this.categoricalBinMap.get(newCCIndex), str);
                if(binNum < 0) {
                    isInvalidValue = true;
                }
            }

            if(isInvalidValue || isMissingValue) {
                binningInfoWritable.setMissingCount(binningInfoWritable.getMissingCount() + 1L);
                binNum = lastBinIndex;
            }

            if(modelConfig.isRegression()) {
                if(posTags.contains(tag)) {
                    binningInfoWritable.getBinCountPos()[binNum] += 1L;
                    binningInfoWritable.getBinWeightPos()[binNum] += weight;
                } else if(negTags.contains(tag)) {
                    binningInfoWritable.getBinCountNeg()[binNum] += 1L;
                    binningInfoWritable.getBinWeightNeg()[binNum] += weight;
                }
            } else {
                // for multiple classification, set bin count to BinCountPos and leave BinCountNeg empty
                binningInfoWritable.getBinCountPos()[binNum] += 1L;
                binningInfoWritable.getBinWeightPos()[binNum] += weight;
            }
        } else if(columnConfig.isNumerical()) {
            int lastBinIndex = binningInfoWritable.getBinBoundaries().size();
            double douVal = 0.0;
            if(units[columnIndex] == null || units[columnIndex].length() == 0) {
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
                binningInfoWritable.setMissingCount(binningInfoWritable.getMissingCount() + 1L);
                if(modelConfig.isRegression()) {
                    if(posTags.contains(tag)) {
                        binningInfoWritable.getBinCountPos()[lastBinIndex] += 1L;
                        binningInfoWritable.getBinWeightPos()[lastBinIndex] += weight;
                    } else if(negTags.contains(tag)) {
                        binningInfoWritable.getBinCountNeg()[lastBinIndex] += 1L;
                        binningInfoWritable.getBinWeightNeg()[lastBinIndex] += weight;
                    }
                }
            } else {
                // For invalid or missing values, no need update sum, squaredSum, max, min ...
                int binNum = getBinNum(binningInfoWritable.getBinBoundaries(), units[columnIndex]);
                if(binNum == -1) {
                    throw new RuntimeException("binNum should not be -1 to this step.");
                }
                if(modelConfig.isRegression()) {
                    if(posTags.contains(tag)) {
                        binningInfoWritable.getBinCountPos()[binNum] += 1L;
                        binningInfoWritable.getBinWeightPos()[binNum] += weight;
                    } else if(negTags.contains(tag)) {
                        binningInfoWritable.getBinCountNeg()[binNum] += 1L;
                        binningInfoWritable.getBinWeightNeg()[binNum] += weight;
                    }
                }
                binningInfoWritable.setSum(binningInfoWritable.getSum() + douVal);
                double squaredVal = douVal * douVal;
                binningInfoWritable.setSquaredSum(binningInfoWritable.getSquaredSum() + squaredVal);
                binningInfoWritable.setTripleSum(binningInfoWritable.getTripleSum() + squaredVal * douVal);
                binningInfoWritable.setQuarticSum(binningInfoWritable.getQuarticSum() + squaredVal * squaredVal);

                if(Double.compare(binningInfoWritable.getMax(), douVal) < 0) {
                    binningInfoWritable.setMax(douVal);
                }
                if(Double.compare(binningInfoWritable.getMin(), douVal) > 0) {
                    binningInfoWritable.setMin(douVal);
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
        return CommonUtils.getBinIndex(binBoundaryList, dval);
    }

    public static int getBinNum(List<Double> binBoundaryList, double dVal) {
        return CommonUtils.getBinIndex(binBoundaryList, dVal);
    }

    private int quickLocateCategoricalBin(Map<String, Integer> map, String val) {
        Integer binNum = map.get(val);
        return ((binNum == null) ? -1 : binNum);
    }

    /**
     * Write column info to reducer for merging.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        LOG.debug("Column binning info: {}", this.columnBinningInfo);
        LOG.debug("Column count info: {}", this.variableCountMap);

        for(Map.Entry<Integer, BinningInfoWritable> entry: this.columnBinningInfo.entrySet()) {
            CountAndFrequentItems cfi = this.variableCountMap.get(entry.getKey());
            if(cfi != null) {
                entry.getValue().setCfiw(
                        new CountAndFrequentItemsWritable(cfi.getCount(), cfi.getInvalidCount(),
                                cfi.getValidNumCount(), cfi.getHyper().getBytes(), cfi.getFrequentItems()));
            } else {
                entry.getValue().setEmpty(true);
                LOG.warn("cci is null for column {}", entry.getKey());
            }

            this.outputKey.set(entry.getKey());
            context.write(this.outputKey, entry.getValue());
        }
    }
}
