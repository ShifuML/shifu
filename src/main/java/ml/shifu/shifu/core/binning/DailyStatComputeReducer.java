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

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;
import ml.shifu.shifu.core.autotype.CountAndFrequentItemsWritable;
import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.core.binning.obj.CategoricalBinInfo;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Collect all statistics together in reducer.
 * 
 * <p>
 * The same format with previous output to make sure consistent with output processing functions.
 * 
 * <p>
 * Only one reducer to make sure all info can be collected together. One reducer is not bottleneck as some times we only
 * have thousands of variables.
 */
public class DailyStatComputeReducer extends Reducer<Text, DailyStatInfoWritable, NullWritable, Text> {

    private final static Logger LOG = LoggerFactory.getLogger(DailyStatComputeReducer.class);

    private static final double EPS = 1e-6;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Prevent too many new objects for output key.
     */
    private Text outputValue;

    /**
     * To concat output string
     */
    private StringBuilder sb = new StringBuilder(2000);

    /**
     * To format double value.
     */
    private DecimalFormat df = new DecimalFormat("##.######");

    private boolean statsExcludeMissingValue;

    private int maxCateSize;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
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
     * Do initialization like ModelConfig and ColumnConfig loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        this.maxCateSize = context.getConfiguration().getInt(Constants.SHIFU_MAX_CATEGORY_SIZE,
                Constants.MAX_CATEGORICAL_BINC_COUNT);

        loadConfigFiles(context);

        this.statsExcludeMissingValue = context.getConfiguration().getBoolean(Constants.SHIFU_STATS_EXLCUDE_MISSING,
                true);

        this.outputValue = new Text();
    }

    @Override
    protected void reduce(Text key, Iterable<DailyStatInfoWritable> values, Context context)
            throws IOException, InterruptedException {
        long start = System.currentTimeMillis();

        Map<String, DailyStatInfoWritable.VariableStatInfo> result = new HashMap<>();

        LOG.info("reduce key." + key);
        //Merge into result
        for(DailyStatInfoWritable info: values) {
            if(info.isEmpty() || MapUtils.isEmpty(info.getVariableDailyStatInfo())) {
                // mapper has no stats, skip it
                continue;
            }
            LOG.info("Daily status info size." + info.getVariableDailyStatInfo().size());
            for(Map.Entry<String, DailyStatInfoWritable.VariableStatInfo> entry : info.getVariableDailyStatInfo().entrySet()){
                DailyStatInfoWritable.VariableStatInfo variableStatInfo = result.get(entry.getKey());
                LOG.info("Init." + entry.getKey() + " " + variableStatInfo);
                if(variableStatInfo == null){
                    variableStatInfo = new DailyStatInfoWritable.VariableStatInfo();
                    result.put(entry.getKey(), variableStatInfo);
                }
                DailyStatInfoWritable.VariableStatInfo statInfo = entry.getValue();
                variableStatInfo.setColumnConfigIndex(statInfo.getColumnConfigIndex());
                ColumnConfig columnConfig = this.columnConfigList.get(statInfo.getColumnConfigIndex());
                variableStatInfo.setTotalCount(variableStatInfo.getTotalCount() + statInfo.getTotalCount());
                variableStatInfo.setMissingCount(variableStatInfo.getMissingCount() + statInfo.getMissingCount());
                variableStatInfo.setSum(variableStatInfo.getSum() + statInfo.getSum());
                variableStatInfo.setSquaredSum(variableStatInfo.getSquaredSum() + statInfo.getSquaredSum());
                variableStatInfo.setTripleSum(variableStatInfo.getTripleSum() + statInfo.getTripleSum());
                variableStatInfo.setQuarticSum(variableStatInfo.getQuarticSum() + statInfo.getQuarticSum());
                if(Double.compare(variableStatInfo.getMax(), statInfo.getMax()) < 0) {
                    variableStatInfo.setMax(statInfo.getMax());
                }

                if(Double.compare(variableStatInfo.getMin(), statInfo.getMin()) > 0) {
                    variableStatInfo.setMin(statInfo.getMin());
                }

                int binSize = 0;
                if(columnConfig.isNumerical() && CollectionUtils.isNotEmpty(columnConfig.getBinBoundary())) {
                    binSize = columnConfig.getBinBoundary().size();
                    variableStatInfo.setBinCountPos(new long[binSize + 1]);
                    variableStatInfo.setBinCountNeg(new long[binSize + 1]);
                    variableStatInfo.setBinWeightPos(new double[binSize + 1]);
                    variableStatInfo.setBinWeightNeg(new double[binSize + 1]);
                    variableStatInfo.setBinCountTotal(new long[binSize + 1]);
                } else if(columnConfig.isCategorical() && CollectionUtils.isNotEmpty(columnConfig.getBinCategory())) {
                    binSize = columnConfig.getBinCategory().size();
                    variableStatInfo.setBinCountPos(new long[binSize + 1]);
                    variableStatInfo.setBinCountNeg(new long[binSize + 1]);
                    variableStatInfo.setBinWeightPos(new double[binSize + 1]);
                    variableStatInfo.setBinWeightNeg(new double[binSize + 1]);
                    variableStatInfo.setBinCountTotal(new long[binSize + 1]);
                }

                for(int i = 0; i < (binSize + 1); i++) {
                    variableStatInfo.getBinCountPos()[i] += statInfo.getBinCountPos() == null ? 0 : statInfo.getBinCountPos()[i];
                    variableStatInfo.getBinCountNeg()[i] += statInfo.getBinCountNeg() == null ? 0 : statInfo.getBinCountNeg()[i];
                    variableStatInfo.getBinWeightPos()[i] += statInfo.getBinWeightPos() == null ? 0 : statInfo.getBinWeightPos()[i];
                    variableStatInfo.getBinWeightNeg()[i] += statInfo.getBinWeightNeg() == null ? 0 : statInfo.getBinWeightNeg()[i];

                    variableStatInfo.getBinCountTotal()[i] += statInfo.getBinCountPos() == null ? 0 : statInfo.getBinCountPos()[i];
                    variableStatInfo.getBinCountTotal()[i] += statInfo.getBinCountNeg() == null ? 0 : statInfo.getBinCountNeg()[i];
                }
                LOG.info("reduce variableStatInfo." + variableStatInfo);
            }
        }

        //calculate result
        for (Map.Entry<String, DailyStatInfoWritable.VariableStatInfo> entry: result.entrySet()){
            DailyStatInfoWritable.VariableStatInfo variableStatInfo = entry.getValue();
            ColumnConfig columnConfig = this.columnConfigList.get(variableStatInfo.getColumnConfigIndex());
            LOG.info("Column index:" + variableStatInfo.getColumnConfigIndex());
            //for numerical, need to do special process
            if(columnConfig.isNumerical()) {
                LOG.info("BinBoundary:" + columnConfig.getBinBoundary().size());
                long p25Count = variableStatInfo.getTotalCount() / 4;
                long medianCount = p25Count * 2;
                long p75Count = p25Count * 3;
                int currentCount = 0;
                for(int i = 0; i < columnConfig.getBinBoundary().size(); i++) {

                    double left = getCutoffBoundary(columnConfig.getBinBoundary().get(i), variableStatInfo.getMax(), variableStatInfo.getMin());
                    double right = ((i == columnConfig.getBinBoundary().size() - 1) ?
                            variableStatInfo.getMax() : getCutoffBoundary(columnConfig.getBinBoundary().get(i + 1), variableStatInfo.getMax(), variableStatInfo.getMin()));
                    LOG.info("stats left" + left + ", right" + right + ", currentCount" + currentCount + ", p25Count" + p25Count + ",getBinCountTotal" + variableStatInfo.getBinCountTotal()[i]);
                    if (p25Count >= currentCount && p25Count < currentCount + variableStatInfo.getBinCountTotal()[i]) {
                        variableStatInfo.setP25th(((p25Count - currentCount) / (double) variableStatInfo.getBinCountTotal()[i])
                                * ( right - left) + left);
                    }

                    if (medianCount >= currentCount && medianCount < currentCount + variableStatInfo.getBinCountTotal()[i]) {
                        variableStatInfo.setMedian(((medianCount - currentCount) / (double) variableStatInfo.getBinCountTotal()[i])
                                * ( right - left) + left);
                    }

                    if (p75Count >= currentCount && p75Count < currentCount + variableStatInfo.getBinCountTotal()[i]) {
                        variableStatInfo.setP75th(((p75Count - currentCount) / (double) variableStatInfo.getBinCountTotal()[i])
                                * ( right - left) + left);
                        // when get 75 percentile stop it
                        break;
                    }
                    currentCount += variableStatInfo.getBinCountTotal()[i];
                }
                LOG.info("stats" + variableStatInfo);
            }

            double[] binPosRate;
            if(modelConfig.isRegression()) {
                binPosRate = computePosRate(variableStatInfo.getBinCountPos(), variableStatInfo.getBinCountNeg());
            } else {
                // for multiple classfication, use rate of categories to compute a value
                binPosRate = computeRateForMultiClassfication(variableStatInfo.getBinCountPos());
            }

            //for categorical, need to do special process
            if(columnConfig.isCategorical()) {
                if(columnConfig.getBinCategory().size() > this.maxCateSize) {
                    LOG.warn("Column {} {} with invalid bin category size.", new String(key.getBytes()), columnConfig.getColumnName(),
                            columnConfig.getBinCategory().size());
                    return;
                }
                // recompute such value for categorical variables
                variableStatInfo.setMin(Double.MAX_VALUE);
                variableStatInfo.setMax(Double.MIN_VALUE);
                variableStatInfo.setSum(0d);
                variableStatInfo.setSquaredSum(0d);
                for(int i = 0; i < binPosRate.length; i++) {
                    LOG.info("stats loop cate" + variableStatInfo);
                    if(!Double.isNaN(binPosRate[i])) {
                        if(Double.compare(variableStatInfo.getMax(), binPosRate[i]) < 0) {
                            variableStatInfo.setMax(binPosRate[i]);
                        }

                        if(Double.compare(variableStatInfo.getMin(), binPosRate[i]) > 0) {
                            variableStatInfo.setMin(binPosRate[i]);
                        }
                        long binCount = variableStatInfo.getBinCountPos()[i] + variableStatInfo.getBinCountNeg()[i];
                        variableStatInfo.setSum(binPosRate[i] * binCount);
                        double squaredVal = binPosRate[i] * binPosRate[i];
                        variableStatInfo.setSquaredSum(squaredVal * binCount);
                        variableStatInfo.setTripleSum(squaredVal * binPosRate[i] * binCount);
                        variableStatInfo.setQuarticSum(squaredVal * squaredVal * binCount);
                    }
                }
                LOG.info("stats cate" + variableStatInfo);
            }

            long realCount = this.statsExcludeMissingValue ? (variableStatInfo.getTotalCount() - variableStatInfo.getMissingCount()) : variableStatInfo.getTotalCount();

            variableStatInfo.setMean(variableStatInfo.getSum() / realCount);

            variableStatInfo.setStdDev(Math.sqrt(Math.abs((variableStatInfo.getSquaredSum() - (variableStatInfo.getSum() * variableStatInfo.getSum()) / realCount + EPS) / (realCount - 1))));
            double aStdDev = Math.sqrt(Math.abs((variableStatInfo.getSquaredSum() - (variableStatInfo.getSum() * variableStatInfo.getSum()) / realCount + EPS) / realCount));

            variableStatInfo.setSkewness(ColumnStatsCalculator.computeSkewness(realCount, variableStatInfo.getMean(), aStdDev, variableStatInfo.getSum(), variableStatInfo.getSquaredSum(), variableStatInfo.getTripleSum()));
            variableStatInfo.setKurtosis(ColumnStatsCalculator.computeKurtosis(realCount, variableStatInfo.getMean(), aStdDev, variableStatInfo.getSum(), variableStatInfo.getSquaredSum(), variableStatInfo.getTripleSum(),
                    variableStatInfo.getQuarticSum()));

            if(modelConfig.isRegression()) {
                LOG.info("ColumnCountMetrics." + ColumnStatsCalculator.calculateColumnMetrics(variableStatInfo.getBinCountNeg(), variableStatInfo.getBinCountPos()));
                variableStatInfo.setColumnCountMetrics(ColumnStatsCalculator.calculateColumnMetrics(variableStatInfo.getBinCountNeg(), variableStatInfo.getBinCountPos()));
                variableStatInfo.setColumnWeightMetrics(ColumnStatsCalculator.calculateColumnMetrics(variableStatInfo.getBinWeightNeg(), variableStatInfo.getBinWeightPos()));
            }
            LOG.info("before logout variableStatInfo." + variableStatInfo);
            sb.append(key)
                    // variable name
                    .append(Constants.DEFAULT_DELIMITER).append(entry.getKey())
                    // date
                    .append(Constants.DEFAULT_DELIMITER).append(columnConfig.getColumnType().toString())
                    // column type
                    .append(Constants.DEFAULT_DELIMITER).append(df.format(variableStatInfo.getMax()))
                    // max
                    .append(Constants.DEFAULT_DELIMITER).append(df.format(variableStatInfo.getMin()))
                    // min
                    .append(Constants.DEFAULT_DELIMITER).append(df.format(variableStatInfo.getMean()))
                    // mean
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getMedian())
                    // median value ?
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getTotalCount())
                    // count
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getMissingCount())
                    // missing count
                    .append(Constants.DEFAULT_DELIMITER).append(df.format(variableStatInfo.getStdDev()))
                    // standard deviation
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getMissingCount() * 1.0d / variableStatInfo.getTotalCount())
                    // missing ratio
                    .append(Constants.DEFAULT_DELIMITER)
                    .append(variableStatInfo.getColumnCountMetrics() == null ? "" : variableStatInfo.getColumnCountMetrics().getWoe())
                    // WOE
                    .append(Constants.DEFAULT_DELIMITER)
                    .append(variableStatInfo.getColumnCountMetrics() == null ? "" : df.format(variableStatInfo.getColumnCountMetrics().getKs()))
                    // KS
                    .append(Constants.DEFAULT_DELIMITER)
                    .append(variableStatInfo.getColumnCountMetrics() == null ? "" : df.format(variableStatInfo.getColumnCountMetrics().getIv()))
                    // IV
                    .append(Constants.DEFAULT_DELIMITER)
                    .append(variableStatInfo.getColumnWeightMetrics() == null ? "" : variableStatInfo.getColumnWeightMetrics().getWoe())
                    // weighted WOE
                    .append(Constants.DEFAULT_DELIMITER)
                    .append(variableStatInfo.getColumnWeightMetrics() == null ? "" : variableStatInfo.getColumnWeightMetrics().getKs())
                    // weighted KS
                    .append(Constants.DEFAULT_DELIMITER)
                    .append(variableStatInfo.getColumnWeightMetrics() == null ? "" : variableStatInfo.getColumnWeightMetrics().getIv())
                    // weighted IV
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getSkewness()) // skewness
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getKurtosis()) // kurtosis

                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getP25th()) // the 25 percentile value
                    .append(Constants.DEFAULT_DELIMITER).append(variableStatInfo.getP75th());

            LOG.info("Output str:"+ sb.toString());
            outputValue.set(sb.toString());
            context.write(NullWritable.get(), outputValue);
            sb.delete(0, sb.length());
        }

        LOG.debug("Time:{}", (System.currentTimeMillis() - start));
    }

    private double[] computePosRate(long[] binCountPos, long[] binCountNeg) {
        assert binCountPos != null && binCountNeg != null && binCountPos.length == binCountNeg.length;
        double[] posRate = new double[binCountPos.length];
        for(int i = 0; i < posRate.length; i++) {
            if(Double.compare(binCountPos[i] + binCountNeg[i], 0d) != 0) {
                // only compute effective pos rate, if /0, don't do it
                posRate[i] = binCountPos[i] * 1.0d / (binCountPos[i] + binCountNeg[i]);
            }
        }
        return posRate;
    }

    private double[] computeRateForMultiClassfication(long[] binCount) {
        double[] rate = new double[binCount.length];
        double sum = 0d;
        for(int i = 0; i < binCount.length; i++) {
            sum += binCount[i];
        }
        for(int i = 0; i < binCount.length; i++) {
            if(Double.compare(sum, 0d) != 0) {
                rate[i] = binCount[i] * 1.0d / sum;
            }
        }
        return rate;
    }

    private double getCutoffBoundary(double val, double max, double min) {
        if ( val == Double.POSITIVE_INFINITY ) {
            return max;
        } else if ( val == Double.NEGATIVE_INFINITY ) {
            return min;
        } else {
            return val;
        }
    }
}
