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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;
import ml.shifu.shifu.core.autotype.CountAndFrequentItemsWritable;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

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
public class UpdateBinningInfoReducer extends Reducer<IntWritable, BinningInfoWritable, NullWritable, Text> {

    private final static Logger LOG = LoggerFactory.getLogger(UpdateBinningInfoReducer.class);

    private static final int MAX_CATEGORICAL_BINC_COUNT = 5000;

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

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
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
     * Do initialization like ModelConfig and ColumnConfig loading.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.statsExcludeMissingValue = context.getConfiguration().getBoolean(Constants.SHIFU_STATS_EXLCUDE_MISSING,
                true);

        this.outputValue = new Text();
    }

    @Override
    protected void reduce(IntWritable key, Iterable<BinningInfoWritable> values, Context context) throws IOException,
            InterruptedException {
        long start = System.currentTimeMillis();
        double sum = 0d;
        double squaredSum = 0d;
        double tripleSum = 0d;
        double quarticSum = 0d;

        long count = 0L, missingCount = 0L;
        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        List<Double> binBoundaryList = null;
        List<String> binCategories = null;
        long[] binCountPos = null;
        long[] binCountNeg = null;
        double[] binWeightPos = null;
        double[] binWeightNeg = null;

        ColumnConfig columnConfig = this.columnConfigList.get(key.get());

        HyperLogLogPlus hyperLogLogPlus = null;
        Set<String> fis = new HashSet<String>();
        long totalCount = 0, invalidCount = 0, validNumCount = 0;
        int binSize = 0;
        for(BinningInfoWritable info: values) {
            CountAndFrequentItemsWritable cfiw = info.getCfiw();
            totalCount += cfiw.getCount();
            invalidCount += cfiw.getInvalidCount();
            validNumCount += cfiw.getValidNumCount();
            fis.addAll(cfiw.getFrequetItems());
            if(hyperLogLogPlus == null) {
                hyperLogLogPlus = HyperLogLogPlus.Builder.build(cfiw.getHyperBytes());
            } else {
                try {
                    hyperLogLogPlus = (HyperLogLogPlus) hyperLogLogPlus.merge(HyperLogLogPlus.Builder.build(cfiw
                            .getHyperBytes()));
                } catch (CardinalityMergeException e) {
                    throw new RuntimeException(e);
                }
            }

            if(info.isNumeric() && binBoundaryList == null) {
                binBoundaryList = info.getBinBoundaries();
                binSize = binBoundaryList.size();
                binCountPos = new long[binSize + 1];
                binCountNeg = new long[binSize + 1];
                binWeightPos = new double[binSize + 1];
                binWeightNeg = new double[binSize + 1];
            }
            if(!info.isNumeric() && binCategories == null) {
                binCategories = info.getBinCategories();
                binSize = binCategories.size();
                binCountPos = new long[binSize + 1];
                binCountNeg = new long[binSize + 1];
                binWeightPos = new double[binSize + 1];
                binWeightNeg = new double[binSize + 1];
            }
            count += info.getTotalCount();
            missingCount += info.getMissingCount();
            // for numeric, such sums are OK, for categorical, such values are all 0, should be updated by using
            // binCountPos and binCountNeg
            sum += info.getSum();
            squaredSum += info.getSquaredSum();
            tripleSum += info.getTripleSum();
            quarticSum += info.getQuarticSum();
            if(Double.compare(max, info.getMax()) < 0) {
                max = info.getMax();
            }

            if(Double.compare(min, info.getMin()) > 0) {
                min = info.getMin();
            }

            for(int i = 0; i < (binSize + 1); i++) {
                binCountPos[i] += info.getBinCountPos()[i];
                binCountNeg[i] += info.getBinCountNeg()[i];
                binWeightPos[i] += info.getBinWeightPos()[i];
                binWeightNeg[i] += info.getBinWeightNeg()[i];
            }
        }

        double[] binPosRate;
        if(modelConfig.isRegression()) {
            binPosRate = computePosRate(binCountPos, binCountNeg);
        } else {
            // for multiple classfication, use rate of categories to compute a value
            binPosRate = computeRateForMultiClassfication(binCountPos);
        }
        String binBounString = null;

        if(columnConfig.isCategorical()) {
            if(binCategories.size() < 0 || binCategories.size() > MAX_CATEGORICAL_BINC_COUNT) {
                LOG.warn("Column {} {} with invalid bin category size.", key.get(), columnConfig.getColumnName(),
                        binCategories.size());
                return;
            }
            binBounString = Base64Utils.base64Encode("["
                    + StringUtils.join(binCategories, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR) + "]");
            // recompute such value for categorical variables
            min = Double.MAX_VALUE;
            max = Double.MIN_VALUE;
            sum = 0d;
            squaredSum = 0d;
            for(int i = 0; i < binPosRate.length; i++) {
                if(!Double.isNaN(binPosRate[i])) {
                    if(Double.compare(max, binPosRate[i]) < 0) {
                        max = binPosRate[i];
                    }

                    if(Double.compare(min, binPosRate[i]) > 0) {
                        min = binPosRate[i];
                    }
                    long binCount = binCountPos[i] + binCountNeg[i];
                    sum += binPosRate[i] * binCount;
                    double squaredVal = binPosRate[i] * binPosRate[i];
                    squaredSum += squaredVal * binCount;
                    tripleSum += squaredVal * binPosRate[i] * binCount;
                    quarticSum += squaredVal * squaredVal * binCount;
                }
            }
        } else {
            if(binBoundaryList.size() == 0) {
                LOG.warn("Column {} {} with invalid bin boundary size.", key.get(), columnConfig.getColumnName(),
                        binBoundaryList.size());
                return;
            }
            binBounString = binBoundaryList.toString();
        }

        ColumnMetrics columnCountMetrics = null;
        ColumnMetrics columnWeightMetrics = null;
        if(modelConfig.isRegression()) {
            columnCountMetrics = ColumnStatsCalculator.calculateColumnMetrics(binCountNeg, binCountPos);
            columnWeightMetrics = ColumnStatsCalculator.calculateColumnMetrics(binWeightNeg, binWeightPos);
        }

        // To make it be consistent with SPDT, missingCount is excluded to compute mean, stddev ...
        long realCount = this.statsExcludeMissingValue ? (count - missingCount) : count;

        double mean = sum / realCount;
        double stdDev = Math.sqrt(Math.abs((squaredSum - (sum * sum) / realCount + EPS) / (realCount - 1)));
        double aStdDev = Math.sqrt(Math.abs((squaredSum - (sum * sum) / realCount + EPS) / realCount));

        double skewness = ColumnStatsCalculator.computeSkewness(realCount, mean, aStdDev, sum, squaredSum, tripleSum);
        double kurtosis = ColumnStatsCalculator.computeKurtosis(realCount, mean, aStdDev, sum, squaredSum, tripleSum,
                quarticSum);

        sb.append(key.get())
                .append(Constants.DEFAULT_DELIMITER)
                .append(binBounString)
                .append(Constants.DEFAULT_DELIMITER)
                .append(Arrays.toString(binCountNeg))
                .append(Constants.DEFAULT_DELIMITER)
                .append(Arrays.toString(binCountPos))
                .append(Constants.DEFAULT_DELIMITER)
                .append(Arrays.toString(new double[0]))
                .append(Constants.DEFAULT_DELIMITER)
                .append(Arrays.toString(binPosRate))
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnCountMetrics == null ? "" : df.format(columnCountMetrics.getKs()))
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnWeightMetrics == null ? "" : df.format(columnWeightMetrics.getIv()))
                .append(Constants.DEFAULT_DELIMITER)
                .append(df.format(max))
                .append(Constants.DEFAULT_DELIMITER)
                .append(df.format(min))
                .append(Constants.DEFAULT_DELIMITER)
                .append(df.format(mean))
                .append(Constants.DEFAULT_DELIMITER)
                .append(df.format(stdDev))
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnConfig.isCategorical() ? "C" : "N")
                .append(Constants.DEFAULT_DELIMITER)
                .append(df.format(mean))
                .append(Constants.DEFAULT_DELIMITER)
                .append(missingCount)
                .append(Constants.DEFAULT_DELIMITER)
                .append(count)
                .append(Constants.DEFAULT_DELIMITER)
                .append(missingCount * 1.0d / count)
                .append(Constants.DEFAULT_DELIMITER)
                .append(Arrays.toString(binWeightNeg))
                .append(Constants.DEFAULT_DELIMITER)
                .append(Arrays.toString(binWeightPos))
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnCountMetrics == null ? "" : columnCountMetrics.getWoe())
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnWeightMetrics == null ? "" : columnWeightMetrics.getWoe())
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnWeightMetrics == null ? "" : columnWeightMetrics.getKs())
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnCountMetrics == null ? "" : columnCountMetrics.getIv())
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnCountMetrics == null ? Arrays.toString(new double[binSize + 1]) : columnCountMetrics
                        .getBinningWoe().toString())
                .append(Constants.DEFAULT_DELIMITER)
                .append(columnWeightMetrics == null ? Arrays.toString(new double[binSize + 1]) : columnWeightMetrics
                        .getBinningWoe().toString()).append(Constants.DEFAULT_DELIMITER).append(skewness)
                .append(Constants.DEFAULT_DELIMITER).append(kurtosis).append(Constants.DEFAULT_DELIMITER)
                .append(totalCount).append(Constants.DEFAULT_DELIMITER).append(invalidCount)
                .append(Constants.DEFAULT_DELIMITER).append(validNumCount).append(Constants.DEFAULT_DELIMITER)
                .append(hyperLogLogPlus.cardinality()).append(Constants.DEFAULT_DELIMITER)
                .append(limitedFrequentItems(fis));

        outputValue.set(sb.toString());
        context.write(NullWritable.get(), outputValue);
        sb.delete(0, sb.length());
        LOG.debug("Time:{}", (System.currentTimeMillis() - start));
    }

    private static String limitedFrequentItems(Set<String> fis) {
        StringBuilder sb = new StringBuilder(200);
        int size = Math.min(fis.size(), CountAndFrequentItemsWritable.FREQUET_ITEM_MAX_SIZE * 10);
        Iterator<String> iterator = fis.iterator();
        int i = 0;
        while(i < size) {
            String next = iterator.next().replaceAll("\\" + Constants.DEFAULT_DELIMITER, " ").replace(",", " ");
            sb.append(next);
            if(i != size - 1) {
                sb.append(",");
            }
            i += 1;
        }
        return sb.toString();
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
}
