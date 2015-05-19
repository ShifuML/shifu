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

import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.util.CommonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Normalizer
 * </p>
 * formula:
 * </p> <code>norm_result = (value - means) / stdev</code> </p>
 * <p/>
 * The stdDevCutOff should be setting, by default it's 4
 * </p>
 * <p/>
 * The <code>value</code> should less than mean + stdDevCutOff * stdev
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
     * @param config
     *            - @ColumnConfig to create normalizer
     */
    public Normalizer(ColumnConfig config) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     * stdDevCutOff will be STD_DEV_CUTOFF
     * 
     * @param config
     *            - @ColumnConfig to create normalizer
     * @param method
     *            - NormalizMethod to use
     */
    public Normalizer(ColumnConfig config, NormalizeMethod method) {
        this(config, method, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     * NormalizeMethod method will be NormalizeMethod.ZScore
     * 
     * @param config
     *            - @ColumnConfig to create normalizer
     * @param cutoff
     *            - stand_dev_cutoff to use
     */
    public Normalizer(ColumnConfig config, Double cutoff) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     * NormalizeMethod method will be NormalizeMethod.ZScore
     * 
     * @param config
     *            - @ColumnConfig to create normalizer
     * @param method
     *            - NormalizMethod to use
     * @param cutoff
     *            - stand_dev_cutoff to use
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
        switch(method) {
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
     * @param config
     *            - @ColumnConfig to normalize data
     * @param raw
     *            - raw input data
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw) {
        return normalize(config, raw, NormalizeMethod.ZScore);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info and normalized method
     * 
     * @param config
     *            - @ColumnConfig to normalize data
     * @param raw
     *            - raw input data
     * @param method
     *            - the method used to do normalization
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, NormalizeMethod method) {
        return normalize(config, raw, method, STD_DEV_CUTOFF);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info and standard deviation cutoff
     * 
     * @param config
     *            - @ColumnConfig to normalize data
     * @param raw
     *            - raw input data
     * @param stdDevCutoff
     *            - the standard deviation cutoff to use
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, double stdDevCutoff) {
        return normalize(config, raw, NormalizeMethod.ZScore, stdDevCutoff);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info, normalized method and standard deviation cutoff
     * 
     * @param config
     *            - @ColumnConfig to normalize data
     * @param raw
     *            - raw input data
     * @param method
     *            - the method used to do normalization
     * @param stdDevCutoff
     *            - the standard deviation cutoff to use
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, NormalizeMethod method, double stdDevCutoff) {
        if(method == null) {
            method = NormalizeMethod.ZScore;
        }

        switch(method) {
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
     * @param config
     *            - @ColumnConfig info
     * @param raw
     *            - input column value
     * @return - normalized value for MaxMin method
     */
    private static Double getMaxMinScore(ColumnConfig config, String raw) {
        if(config.isCategorical()) {
            // TODO, doesn't support
        } else {
            Double value = Double.parseDouble(raw);
            return (value - config.getColumnStats().getMin())
                    / (config.getColumnStats().getMax() - config.getColumnStats().getMin());
        }
        return null;
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     * 
     * @param config
     *            - @ColumnConfig info
     * @param raw
     *            - input column value
     * @param cutoff
     *            - standard deviation cut off
     * @return - normalized value for ZScore method
     */
    private static Double getZScore(ColumnConfig config, String raw, Double cutoff) {
        Double stdDevCutOff;
        if(cutoff != null && !cutoff.isInfinite() && !cutoff.isNaN()) {
            stdDevCutOff = cutoff;
        } else {
            stdDevCutOff = STD_DEV_CUTOFF;
        }

        double value = parseRawValue(config, raw);
        return computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff);
    }
    
    /**
     * Calculate desiring value parsed from raw based on ColumnConfig.
     * 
     * @param config
     *            - @ColumnConfig info
     * @param raw
     *            - input column value
     * @return desiring value parsed from raw. For categorical type, return BinPosRate. For numerical type, return 
     *         corresponding double value. For missing data, return default value using 
     *         {@link Normalizer#defaultMissingValue}.
     */
    private static double parseRawValue(ColumnConfig config, String raw) {
        double value = 0.0;
        if(config.isCategorical()) {
            int index = config.getBinCategory().indexOf(raw);
            if(index == -1) {
                // When raw is not a invalid Category, a proper value should be returned.
                value = defaultMissingValue(config);
            } else {
                value = config.getBinPosRate().get(index);
            }
        } else {
            try {
                value = Double.parseDouble(raw);
            } catch (Exception e) {
                log.debug("Not decimal format " + raw + ", using default!");
                // When raw is non-parsable, usually we use mean value instead.
                // So the corresponding zscore becomes 0.
                value = defaultMissingValue(config);
            }
        }
        
        return value;
    }
    
    /**
     * Get the default value for missing data. 
     * 
     * @param config
     *            - @ColumnConfig info
     * @return - default value for missing data. Now simply return Mean value.
     */
    private static double defaultMissingValue(ColumnConfig config) {
        // TODO
        // Here simply return Mean as the default value. We may customize it later.
        return config.getMean();
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     * 
     * @param config
     *            - @ColumnConfig info
     * @param raw
     *            - input column value
     * @param cutoff
     *            - standard deviation cut off
     * @return - normalized value for ZScore method
     */
    public static Double zScoreNormalize(ColumnConfig config, String raw, double cutoff) {
        return getZScore(config, raw, cutoff);
    }
    
    /**
     * Compute the normalized data for Woe Score.
     * 
     * @param config
     *            - @ColumnConfig info
     * @param raw
     *            - input column value
     * @param isWeightedNorm
     *            - if use weighted woe
     * @return - normalized value for Woe method. For missing value, we return the value in last bin. Since the last
     *           bin refers to the missing value bin. 
     */
    public static Double woeNormalize(ColumnConfig config, String raw, boolean isWeightNorm) {
        List<Double> woeBins = isWeightNorm ? config.getBinWeightedWoe() : config.getBinCountWoe();
        int binIndex = CommonUtils.getBinNum(config, raw);
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            return woeBins.get(woeBins.size() - 1);
        } else {
            return woeBins.get(binIndex);
        }
    }
    
    /**
     * Compute the normalized data for hbrid normalize. Use zscore noramlize for numerical data. Use woe normalize
     * for categorical data.
     * 
     * @param config
     *            - @ColumnConfig info
     * @param raw
     *            - input column value
     * @param cutoff
     *            - standard deviation cut off
     * @param isWeightedNorm
     *            - if use weighted woe
     * @return - normalized value for hybrid method.
     */
    public static Double hybridNormalize(ColumnConfig config, String raw, Double cutoff, boolean isWeightedNorm) {
        Double normValue;
        if (config.isNumerical()) {
            // For numerical data, use zscore.
            normValue = getZScore(config, raw, cutoff);
        } else {
            // For categorical data, use woe.
            normValue = woeNormalize(config, raw, isWeightedNorm);
        }
        
        return normValue;
    }

    /**
     * Compute the zscore, by original value, mean, standard deviation and standard deviation cutoff
     * 
     * @param var
     *            - original value
     * @param mean
     *            - mean value
     * @param stdDev
     *            - standard deviation
     * @param stdDevCutOff
     *            - standard deviation cutoff
     * @return zscore
     */
    public static double computeZScore(double var, double mean, double stdDev, double stdDevCutOff) {
        double maxCutOff = mean + stdDevCutOff * stdDev;
        if(var > maxCutOff) {
            var = maxCutOff;
        }

        double minCutOff = mean - stdDevCutOff * stdDev;
        if(var < minCutOff) {
            var = minCutOff;
        }

        if(stdDev > 0.00001) {
            return (var - mean) / stdDev;
        } else {
            return 0.0;
        }
    }

}
