/*
 * Copyright [2012-2015] eBay Software Foundation
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

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;
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

    private static final int MAX_CATEGORICAL_BINC_COUNT = 4000;

    @SuppressWarnings("unused")
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

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
    private void loadConfigFiles(final Context context) {
        try {
            SourceType sourceType = SourceType.valueOf(context.getConfiguration().get(
                    Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
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
        this.outputValue = new Text();
    }

    @Override
    protected void reduce(IntWritable key, Iterable<BinningInfoWritable> values, Context context) throws IOException,
            InterruptedException {
        long start = System.currentTimeMillis();
        double sum = 0d;
        double sumSquare = 0d;
        long count = 0L, missingCount = 0L;
        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        List<Double> binBoundaryList = null;
        List<String> binCategories = null;
        long[] binCountPos = null;
        long[] binCountNeg = null;
        double[] binWeightPos = null;
        double[] binWeightNeg = null;

        ColumnConfig columnConfig = this.columnConfigList.get(key.get());

        int binSize = 0;
        for(BinningInfoWritable info: values) {
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
            sum += info.getSum();
            sumSquare += info.getSquaredSum();
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

        double[] binPosRate = computePosRate(binCountPos, binCountNeg);

        String binBounString = null;
        if(columnConfig.isCategorical()) {
            if(binCategories.size() == 0 || binCategories.size() > MAX_CATEGORICAL_BINC_COUNT) {
                LOG.warn("Column {} {} with invalid bin boundary size.", key.get(), columnConfig.getColumnName(),
                        binCategories.size());
                return;
            }
            binBounString = Base64Utils.base64Encode("["
                    + StringUtils.join(binCategories, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR) + "]");
            // recompute such value for categorial variables
            min = Double.MAX_VALUE;
            max = Double.MIN_VALUE;
            sum = 0d;
            sumSquare = 0d;
            for(int i = 0; i < binPosRate.length; i++) {
                if(Double.compare(max, binPosRate[i]) < 0) {
                    max = binPosRate[i];
                }

                if(Double.compare(min, binPosRate[i]) > 0) {
                    min = binPosRate[i];
                }
                sum += binPosRate[i] * (binCountPos[i] + binCountNeg[i]);
                sumSquare += binPosRate[i] * binPosRate[i] * (binCountPos[i] + binCountNeg[i]);
            }
        } else {
            if(binBoundaryList.size() == 0) {
                LOG.warn("Column {} {} with invalid bin boundary size.", key.get(), columnConfig.getColumnName(),
                        binBoundaryList.size());
                return;
            }
            binBounString = binBoundaryList.toString();
        }

        ColumnMetrics columnCountMetrics = ColumnStatsCalculator.calculateColumnMetrics(binCountNeg, binCountPos);

        ColumnMetrics columnWeightMetrics = ColumnStatsCalculator.calculateColumnMetrics(binWeightNeg, binWeightPos);

        sb.append(key.get()).append(Constants.DEFAULT_DELIMITER).append(binBounString)
                .append(Constants.DEFAULT_DELIMITER).append(Arrays.toString(binCountNeg))
                .append(Constants.DEFAULT_DELIMITER).append(Arrays.toString(binCountPos))
                .append(Constants.DEFAULT_DELIMITER).append(Arrays.toString(new double[0]))
                .append(Constants.DEFAULT_DELIMITER).append(Arrays.toString(binPosRate))
                .append(Constants.DEFAULT_DELIMITER).append(df.format(columnCountMetrics.getKs()))
                .append(Constants.DEFAULT_DELIMITER).append(df.format(columnWeightMetrics.getIv()))
                .append(Constants.DEFAULT_DELIMITER).append(df.format(max)).append(Constants.DEFAULT_DELIMITER)
                .append(df.format(min)).append(Constants.DEFAULT_DELIMITER).append(df.format(sum / count))
                .append(Constants.DEFAULT_DELIMITER)
                .append(df.format(Math.sqrt(Math.abs(((sumSquare / count) - power2(sum / count))))))
                .append(Constants.DEFAULT_DELIMITER).append(columnConfig.isCategorical() ? "C" : "N")
                .append(Constants.DEFAULT_DELIMITER).append(df.format(sum / count)).append(Constants.DEFAULT_DELIMITER)
                .append(missingCount).append(Constants.DEFAULT_DELIMITER).append(count)
                .append(Constants.DEFAULT_DELIMITER).append(missingCount * 1.0d / count)
                .append(Constants.DEFAULT_DELIMITER).append(Arrays.toString(binWeightNeg))
                .append(Constants.DEFAULT_DELIMITER).append(Arrays.toString(binWeightPos))
                .append(Constants.DEFAULT_DELIMITER).append(columnCountMetrics.getWoe())
                .append(Constants.DEFAULT_DELIMITER).append(columnWeightMetrics.getWoe())
                .append(Constants.DEFAULT_DELIMITER).append(columnWeightMetrics.getKs())
                .append(Constants.DEFAULT_DELIMITER).append(columnCountMetrics.getIv())
                .append(Constants.DEFAULT_DELIMITER).append(columnCountMetrics.getBinningWoe().toString())
                .append(Constants.DEFAULT_DELIMITER).append(columnWeightMetrics.getBinningWoe().toString());

        outputValue.set(sb.toString());
        context.write(NullWritable.get(), outputValue);
        sb.delete(0, sb.length());
        LOG.info("Time:{}", (System.currentTimeMillis() - start));
    }

    private double[] computePosRate(long[] binCountPos, long[] binCountNeg) {
        assert binCountPos != null && binCountNeg != null && binCountPos.length == binCountNeg.length;
        double[] posRate = new double[binCountPos.length];
        for(int i = 0; i < posRate.length; i++) {
            posRate[i] = binCountPos[i] * 1.0d / (binCountPos[i] + binCountNeg[i]);
        }
        return posRate;
    }

    private double power2(double data) {
        return data * data;
    }

}
