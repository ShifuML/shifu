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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.MissValueFillType;
import ml.shifu.shifu.util.CommonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Normalizer
 * </p>
 * formula:
 * </p>
 * <code>norm_result = (value - means) / stdev</code>
 * </p>
 * <p/>
 * The stdDevCutOff should be setting, by default it's 4
 * </p>
 * <p/>
 * The <code>value</code> should less than  mean + stdDevCutOff * stdev
 * </p>
 * and larger than mean - stdDevCutOff * stdev
 */
public class Normalizer {

    private static Logger log = LoggerFactory.getLogger(Normalizer.class);
    public static final double STD_DEV_CUTOFF = 4.0d;

    public enum NormalizeMethod {
        ZScore, MaxMin;
    }

    private ColumnConfig config;
    private Double stdDevCutOff = 4.0;
    private NormalizeMethod method;

    /**
     * Create @Normalizer, according @ColumnConfig
     * NormalizeMethod method will be NormalizeMethod.ZScore
     * stdDevCutOff will be STD_DEV_CUTOFF
     *
     * @param config - @ColumnConfig to create normalizer
     */
    public Normalizer(ColumnConfig config) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     * stdDevCutOff will be STD_DEV_CUTOFF
     *
     * @param config - @ColumnConfig to create normalizer
     * @param method - NormalizMethod to use
     */
    public Normalizer(ColumnConfig config, NormalizeMethod method) {
        this(config, method, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     * NormalizeMethod method will be NormalizeMethod.ZScore
     *
     * @param config - @ColumnConfig to create normalizer
     * @param cutoff - stand_dev_cutoff to use
     */
    public Normalizer(ColumnConfig config, Double cutoff) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     * NormalizeMethod method will be NormalizeMethod.ZScore
     *
     * @param config - @ColumnConfig to create normalizer
     * @param method - NormalizMethod to use
     * @param cutoff - stand_dev_cutoff to use
     */
    public Normalizer(ColumnConfig config, NormalizeMethod method, Double cutoff) {
        this.config = config;
        this.method = method;
        this.stdDevCutOff = cutoff;
    }

    /**
     * Normalize the input data for column
     *
     * @param raw
     * @return
     */
    public Double normalize(String raw) {
        switch (method) {
            case ZScore:
                return getZScore(config, raw, stdDevCutOff);
            case MaxMin:
                return getMaxMinScore(config, raw);
            default:
                return 0.0;
        }
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info
     *
     * @param config - @ColumnConfig to normalize data
     * @param raw    - raw input data
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw) {
        return normalize(config, raw, NormalizeMethod.ZScore);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info and normalized method
     *
     * @param config - @ColumnConfig to normalize data
     * @param raw    - raw input data
     * @param method - the method used to do normalization
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, NormalizeMethod method) {
        return normalize(config, raw, method, STD_DEV_CUTOFF);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info and standard deviation cutoff
     *
     * @param config       - @ColumnConfig to normalize data
     * @param raw          - raw input data
     * @param stdDevCutoff - the standard deviation cutoff to use
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, double stdDevCutoff) {
        return normalize(config, raw, NormalizeMethod.ZScore, stdDevCutoff);
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     *
     * @param config   - @ColumnConfig info
     * @param raw      - input column value
     * @param cutoff   - standard deviation cut off
     * @param fillType - fill value type for missing value
     * @return - normalized value for ZScore method
     */
    public static Double zScoreNormalize(ColumnConfig config, String raw, double cutoff, MissValueFillType fillType) {
        return getZScore(config, raw, cutoff, fillType);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info, normalized method and standard deviation cutoff
     *
     * @param config       - @ColumnConfig to normalize data
     * @param raw          - raw input data
     * @param method       - the method used to do normalization
     * @param stdDevCutoff - the standard deviation cutoff to use
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, NormalizeMethod method, double stdDevCutoff) {
        if (method == null) {
            method = NormalizeMethod.ZScore;
        }

        switch (method) {
            case ZScore:
                return getZScore(config, raw, stdDevCutoff);
            case MaxMin:
                return getMaxMinScore(config, raw);
            default:
                return 0.0;
        }
    }

    /**
     * Compute the normalized data for @NormalizeMethod.MaxMin
     *
     * @param config - @ColumnConfig info
     * @param raw    - input column value
     * @return - normalized value for MaxMin method
     */
    private static Double getMaxMinScore(ColumnConfig config, String raw) {
        if (config.isCategorical()) {
            //TODO, doesn't support
        } else {
            Double value = Double.parseDouble(raw);
            return (value - config.getColumnStats().getMin()) /
                    (config.getColumnStats().getMax() - config.getColumnStats().getMin());
        }
        return null;
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     *
     * @param config - @ColumnConfig info
     * @param raw    - input column value
     * @param cutoff - standard deviation cut off
     * @return - normalized value for ZScore method
     */
    private static Double getZScore(ColumnConfig config, String raw, Double cutoff) {
        return getZScore(config, raw, cutoff, MissValueFillType.ZERO);
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     *
     * @param config - @ColumnConfig info
     * @param raw    - input column value
     * @param cutoff - standard deviation cut off
     * @param fillType - fill value type for missing value
     * @return - normalized value for ZScore method
     */
    private static Double getZScore(ColumnConfig config, String raw, Double cutoff, MissValueFillType fillType) {
        Double stdDevCutOff;
        if (cutoff != null && !cutoff.isInfinite() && !cutoff.isNaN()) {
            stdDevCutOff = cutoff;
        } else {
            stdDevCutOff = STD_DEV_CUTOFF;
        }
        
        if (config.isCategorical()) {
            // int index = config.getBinCategory().indexOf(raw);
            int index = getCategoryIndexOrMostFreq(config, raw);

            if (index == -1) {
                // Use default value configured by MissValueFillType for missing value.
                // return getDefaultValueByFillType(fillType, config);
                return computeZScore(config.getBinPosRate().get(0), config.getMean(), config.getStdDev(), stdDevCutOff);
            } else {
                return computeZScore(config.getBinPosRate().get(index), config.getMean(), config.getStdDev(), stdDevCutOff);
            }
        } else {
            double value = 0.0;
            try {
                value = Double.parseDouble(raw);
            } catch (Exception e) {
                log.debug("Not decimal format " + raw + ", using default!");
                value = getDefaultValueByFillType(fillType, config);
            }
            
            return computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff);
        }
    }

    /**
     * Get the index of categorical value in bin category list
     * If the column value is not in bin category list, just return the most frequent column value index
     *
     * @param config - @ColumnConfig info
     * @param raw    - input column value
     * @return       - value index in bin category list
     */
    private static int getCategoryIndexOrMostFreq(ColumnConfig config, String raw) {
        int mostFreqIndex = -1;
        int maxValCnt = -1;
        int categoryValSize = config.getBinCategory().size();
        for ( int i = 0; i < categoryValSize; i ++ ) {
            if ( config.getBinCategory().get(i).equals(raw) ) {
                return i;
            }

            int catValCnt = config.getBinCountNeg().get(i) + config.getBinCountPos().get(i);
            if ( catValCnt >  maxValCnt ) {
                maxValCnt = catValCnt;
                mostFreqIndex = i;
            }
        }

        return mostFreqIndex;
    }

    /**
     * Compute the normalized data for Woe Score
     *
     * @param config - @ColumnConfig info
     * @param raw    - input column value
     * @param isWeightedNorm - if use weighted woe
     * @param fillType - fill value type for missing value
     * @return - normalized value for Woe method
     */
    public static Double woeNormalize(ColumnConfig config, String raw, boolean isWeightedNorm, MissValueFillType fillType) {
        // binNum = -1 when val is empty/null/non-parsable.
        // we regard this as missing value, return default value configured by MissValueFillType instead.
        int binNum = CommonUtils.getBinNum(config, raw);
        if (binNum == -1) {
            return getDefaultValueByFillType(fillType, config);
        } else {
            if (isWeightedNorm) {
                return config.getColumnBinning().getBinWeightedWoe().get(binNum);
            } else {
                return config.getColumnBinning().getBinCountWoe().get(binNum);
            }
        }
    }
    
    /**
     * Compute the zscore, by original value, mean, standard deviation and standard deviation cutoff
     *
     * @param var          - original value
     * @param mean         - mean value
     * @param stdDev       - standard deviation
     * @param stdDevCutOff - standard deviation cutoff
     * @return zscore
     */
    public static double computeZScore(double var, double mean, double stdDev, double stdDevCutOff) {
        double maxCutOff = mean + stdDevCutOff * stdDev;
        if (var > maxCutOff) {
            var = maxCutOff;
        }

        double minCutOff = mean - stdDevCutOff * stdDev;
        if (var < minCutOff) {
            var = minCutOff;
        }

        if (stdDev > 0.00001) {
            return (var - mean) / stdDev;
        } else {
            return 0.0;
        }
    }
    
    /**
     * <p>
     * Get default value for missing value.
     * </p>
     * 
     * <p>
     * The Default value choice is based on the configuration setting in modelConfig#normalize#MissValueFillType.
     * The Fill Type can refer to <code>{@link MissValueFillType}</code>.
     * </p>
     * 
     * @param fillType fill type for missing value. @see <code>{@link MissValueFillType}</code>.
     * @param config column config
     * @return Double value used to fill missing value. If the choosing value is null, then return 0.
     */
    public static Double getDefaultValueByFillType(MissValueFillType fillType, ColumnConfig config) {
        Double fillValue = null;
        switch(fillType) {
        case COUNTWOE:
            fillValue = config.getBinCountWoe().get(config.getBinCountWoe().size() - 1);
            break;
        case WEIGHTEDWOE:
            fillValue = config.getBinWeightedWoe().get(config.getBinWeightedWoe().size() - 1);
            break;
        case MEAN:
            fillValue = config.getMean();
            break;            
        case ZERO:
        default:
            return Double.valueOf(0.0);
        }

        return fillValue == null ? Double.valueOf(0.0) : fillValue;
    }
    
}
