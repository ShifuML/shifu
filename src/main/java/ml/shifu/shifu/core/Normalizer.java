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

import java.util.Arrays;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.udf.NormalizeUDF.CategoryMissingNormType;
import ml.shifu.shifu.util.CommonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Util normalization class which is used for any kind of transformation.
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
     * Create @Normalizer, according ColumnConfig
     * NormalizeMethod method will be NormalizeMethod.ZScore
     * stdDevCutOff will be STD_DEV_CUTOFF
     * 
     * @param config
     *            ColumnConfig to create normalizer
     */
    public Normalizer(ColumnConfig config) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according ColumnConfig and NormalizeMethod
     * stdDevCutOff will be STD_DEV_CUTOFF
     * 
     * @param config
     *            ColumnConfig to create normalizer
     * @param method
     *            NormalizMethod to use
     */
    public Normalizer(ColumnConfig config, NormalizeMethod method) {
        this(config, method, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according ColumnConfig and NormalizeMethod
     * NormalizeMethod method will be NormalizeMethod.ZScore
     * 
     * @param config
     *            ColumnConfig to create normalizer
     * @param cutoff
     *            stand_dev_cutoff to use
     */
    public Normalizer(ColumnConfig config, Double cutoff) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according ColumnConfig and NormalizeMethod
     * NormalizeMethod method will be NormalizeMethod.ZScore
     * 
     * @param config
     *            ColumnConfig to create normalizer
     * @param method
     *            NormalizMethod to use
     * @param cutoff
     *            stand_dev_cutoff to use
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
     *            the raw value
     * @return normalized value
     */
    public List<Double> normalize(String raw) {
        return normalize(config, raw, method, stdDevCutOff);
    }

    /**
     * Normalize the raw file, according the ColumnConfig info
     * 
     * @param config
     *            ColumnConfig to normalize data
     * @param raw
     *            raw input data
     * @return normalized value
     */
    public static List<Double> normalize(ColumnConfig config, String raw) {
        return normalize(config, raw, NormalizeMethod.ZScore);
    }

    /**
     * Normalize the raw file, according the ColumnConfig info and normalized method
     * 
     * @param config
     *            ColumnConfig to normalize data
     * @param raw
     *            raw input data
     * @param method
     *            the method used to do normalization
     * @return normalized value
     */
    public static List<Double> normalize(ColumnConfig config, String raw, NormalizeMethod method) {
        return normalize(config, raw, method, STD_DEV_CUTOFF);
    }

    /**
     * Normalize the raw file, according the ColumnConfig info and standard deviation cutoff
     * 
     * @param config
     *            ColumnConfig to normalize data
     * @param raw
     *            raw input data
     * @param stdDevCutoff
     *            the standard deviation cutoff to use
     * @return normalized value
     */
    public static List<Double> normalize(ColumnConfig config, String raw, double stdDevCutoff) {
        return normalize(config, raw, NormalizeMethod.ZScore, stdDevCutoff);
    }

    /**
     * Normalize the raw file, according the ColumnConfig info, normalized method and standard deviation cutoff
     * 
     * @param config
     *            ColumnConfig to normalize data
     * @param raw
     *            raw input data
     * @param method
     *            the method used to do normalization
     * @param stdDevCutoff
     *            the standard deviation cutoff to use
     * @return normalized value
     */
    public static List<Double> normalize(ColumnConfig config, String raw, NormalizeMethod method, double stdDevCutoff) {
        if(method == null) {
            method = NormalizeMethod.ZScore;
        }

        switch(method) {
            case ZScore:
                return zScoreNormalize(config, raw, stdDevCutoff);
            case MaxMin:
                return Arrays.asList(getMaxMinScore(config, raw));
            default:
                return Arrays.asList(new Double[] { 0.0 });
        }
    }

    /**
     * Compute the normalized data for @NormalizeMethod.MaxMin
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @return normalized value for MaxMin method
     */
    private static Double[] getMaxMinScore(ColumnConfig config, String raw) {
        if(config.isCategorical()) {
            // TODO, doesn't support
        } else {
            Double value = Double.parseDouble(raw);
            return new Double[] { (value - config.getColumnStats().getMin())
                    / (config.getColumnStats().getMax() - config.getColumnStats().getMin()) };
        }
        return null;
    }

    /**
     * Normalize the raw data, according the ColumnConfig information and normalization type.
     * Currently, the cutoff value doesn't affect the computation of WOE or WEIGHT_WOE type.
     * 
     * <p>
     * Noticed: currently OLD_ZSCALE and ZSCALE is implemented with the same process method.
     * </p>
     * 
     * @param config
     *            ColumnConfig to normalize data
     * @param raw
     *            raw input data
     * @param cutoff
     *            standard deviation cut off
     * @param type
     *            normalization type of ModelNormalizeConf.NormType
     * @param categoryMissingNormType
     *            missing categorical value norm type
     * @return normalized value. If normType parameter is invalid, then the ZSCALE will be used as default.
     */
    public static List<Double> normalize(ColumnConfig config, String raw, Double cutoff,
            ModelNormalizeConf.NormType type, CategoryMissingNormType categoryMissingNormType) {
        switch(type) {
            case WOE:
                return woeNormalize(config, raw, false);
            case WEIGHT_WOE:
                return woeNormalize(config, raw, true);
            case HYBRID:
                return hybridNormalize(config, raw, cutoff, false);
            case WEIGHT_HYBRID:
                return hybridNormalize(config, raw, cutoff, true);
            case WOE_ZSCORE:
            case WOE_ZSCALE:
                return woeZScoreNormalize(config, raw, cutoff, false);
            case WEIGHT_WOE_ZSCORE:
            case WEIGHT_WOE_ZSCALE:
                return woeZScoreNormalize(config, raw, cutoff, true);
            case ZSCALE_ONEHOT:
                return woeOneHotNormalize(config, raw, cutoff, categoryMissingNormType);
            case DISCRETE_ZSCORE:
            case DISCRETE_ZSCALE:
                return discreteZScoreNormalize(config, raw, cutoff, categoryMissingNormType);
            case OLD_ZSCALE:
            case OLD_ZSCORE:
            case ZSCALE:
            case ZSCORE:
            default:
                return zScoreNormalize(config, raw, cutoff, categoryMissingNormType);
        }
    }

    private static List<Double> woeOneHotNormalize(ColumnConfig config, String raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType) {
        if(config.isNumerical()) {
            return zScoreNormalize(config, raw, cutoff, categoryMissingNormType);
        } else {
            Double[] normVals = new Double[config.getBinCategory().size() + 1];
            Arrays.fill(normVals, 0.0d);

            int binNum = CommonUtils.getBinNum(config, raw);
            if(binNum < 0) {
                binNum = config.getBinCategory().size();
            }
            normVals[binNum] = 1.0d;
            return Arrays.asList(normVals);
        }
    }

    /**
     * Normalize the raw data, according the ColumnConfig information and normalization type.
     * Currently, the cutoff value doesn't affect the computation of WOE or WEIGHT_WOE type.
     * 
     * <p>
     * Noticed: currently OLD_ZSCALE and ZSCALE is implemented with the same process method.
     * </p>
     * 
     * @param config
     *            ColumnConfig to normalize data
     * @param raw
     *            raw input data
     * @param cutoff
     *            standard deviation cut off
     * @param type
     *            normalization type of ModelNormalizeConf.NormType
     * @return normalized value. If normType parameter is invalid, then the ZSCALE will be used as default.
     */
    public static List<Double> normalize(ColumnConfig config, String raw, Double cutoff,
            ModelNormalizeConf.NormType type) {
        return normalize(config, raw, cutoff, type, CategoryMissingNormType.POSRATE);
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param cutoff
     *            standard deviation cut off
     * @param categoryMissingNormType
     *            missing categorical value norm type
     * @return normalized value for ZScore method.
     */
    private static List<Double> zScoreNormalize(ColumnConfig config, String raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType) {
        double stdDevCutOff = checkCutOff(cutoff);
        double value = parseRawValue(config, raw, categoryMissingNormType);
        return Arrays.asList(computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff));
    }

    /**
     * Compute the zscore value after do discreting in each bin for numerical value, for categorical feature, use
     * positive rate.
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param cutoff
     *            standard deviation cut off
     * @param categoryMissingNormType
     *            missing categorical value norm type
     * @return normalized value for ZScore method.
     */
    private static List<Double> discreteZScoreNormalize(ColumnConfig config, String raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType) {
        double stdDevCutOff = checkCutOff(cutoff);
        double value = 0;
        if(config.isCategorical()) {
            value = parseRawValue(config, raw, categoryMissingNormType);
        } else {
            int binIndex = CommonUtils.getBinNum(config, raw);
            if(binIndex < 0 || binIndex >= config.getBinBoundary().size()) {
                // missing value, use mean value, after zscore, it is 0
                value = config.getMean();
            } else {
                List<Double> binBoundaries = config.getBinBoundary();
                if(binIndex == 0) {
                    // the first bin, use min value
                    value = config.getColumnStats().getMin();
                } else {
                    value = binBoundaries.get(binIndex);
                }
            }
        }
        return Arrays.asList(computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff));
    }

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param cutoff
     *            standard deviation cut off
     * @return normalized value for ZScore method.
     */
    private static List<Double> zScoreNormalize(ColumnConfig config, String raw, Double cutoff) {
        double stdDevCutOff = checkCutOff(cutoff);
        double value = parseRawValue(config, raw, CategoryMissingNormType.POSRATE);
        return Arrays.asList(computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff));
    }

    /**
     * Parse raw value based on ColumnConfig.
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param categoryMissingNormType
     *            missing categorical value norm type
     * @return parsed raw value. For categorical type, return BinPosRate. For numerical type, return
     *         corresponding double value. For missing data, return default value using
     *         {@link Normalizer#defaultMissingValue}.
     */
    private static double parseRawValue(ColumnConfig config, String raw, CategoryMissingNormType categoryMissingNormType) {
        if(categoryMissingNormType == null) {
            categoryMissingNormType = CategoryMissingNormType.POSRATE;
        }
        double value = 0.0;
        if(config.isCategorical()) {
            int index = CommonUtils.getBinNum(config, raw);
            if(index == -1) {
                switch(categoryMissingNormType) {
                    case POSRATE:
                        // last one is missing bin, if it is missing, using pos rate for default value.
                        value = config.getBinPosRate().get(config.getBinPosRate().size() - 1);
                        break;
                    case MEAN:
                    default:
                        value = defaultMissingValue(config);
                        break;
                }
            } else {
                Double binPosRate = config.getBinPosRate().get(index);
                if(binPosRate != null) {
                    value = binPosRate.doubleValue();
                } else {
                    switch(categoryMissingNormType) {
                        case POSRATE:
                            // last one is missing bin, if it is missing, using pos rate for default value.
                            value = config.getBinPosRate().get(config.getBinPosRate().size() - 1);
                            break;
                        case MEAN:
                        default:
                            value = defaultMissingValue(config);
                            break;
                    }
                }
            }
        } else {
            try {
                value = Double.parseDouble(raw);
            } catch (Exception e) {
                log.debug("Not decimal format " + raw + ", using default!");
                value = defaultMissingValue(config);
            }
        }

        return value;
    }

    /**
     * Get the default value for missing data.
     * 
     * @param config
     *            ColumnConfig info
     * @return default value for missing data. Now simply return Mean value. If mean is null then return 0.
     */
    public static double defaultMissingValue(ColumnConfig config) {
        // TODO return 0 for mean == null is correct or reasonable?
        return config.getMean() == null ? 0 : config.getMean().doubleValue();
    }

    /**
     * Compute the normalized data for Woe Score.
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param isWeightedNorm
     *            if use weighted woe
     * @return normalized value for Woe method. For missing value, we return the value in last bin. Since the last
     *         bin refers to the missing value bin.
     */
    private static List<Double> woeNormalize(ColumnConfig config, String raw, boolean isWeightedNorm) {
        List<Double> woeBins = isWeightedNorm ? config.getBinWeightedWoe() : config.getBinCountWoe();
        int binIndex = 0;
        if(config.isHybrid()) {
            binIndex = CommonUtils.getCategoicalBinIndex(config.getBinCategory(), raw);
            if(binIndex != -1) {
                binIndex = binIndex + config.getBinBoundary().size(); // append the first numerical bins
            } else {
                double douVal = CommonUtils.parseNumber(raw);
                if(Double.isNaN(douVal)) {
                    binIndex = config.getBinBoundary().size() + config.getBinCategory().size();
                } else {
                    binIndex = CommonUtils.getBinIndex(config.getBinBoundary(), douVal);
                }
            }
        } else {
            binIndex = CommonUtils.getBinNum(config, raw);
        }
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            return Arrays.asList(new Double[] { woeBins.get(woeBins.size() - 1) });
        } else {
            return Arrays.asList(new Double[] { woeBins.get(binIndex) });
        }
    }

    /**
     * Compute the normalized value for woe zscore normalize.Take woe as variable value and using zscore normalizing
     * to compute zscore of woe.
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param cutoff
     *            standard deviation cut off
     * @param isWeightedNorm
     *            if use weighted woe
     * @return normalized value for woe zscore method.
     */
    private static List<Double> woeZScoreNormalize(ColumnConfig config, String raw, Double cutoff,
            boolean isWeightedNorm) {
        double stdDevCutOff = checkCutOff(cutoff);
        double woe = woeNormalize(config, raw, isWeightedNorm).get(0);
        // TODO cache such computing to avoid computing each time
        double[] meanAndStdDev = calculateWoeMeanAndStdDev(config, isWeightedNorm);
        return Arrays.asList(computeZScore(woe, meanAndStdDev[0], meanAndStdDev[1], stdDevCutOff));
    }

    /**
     * Compute the normalized data for hbrid normalize. Use zscore noramlize for numerical data. Use woe normalize
     * for categorical data while use weight woe normalize when isWeightedNorm is true.
     * 
     * @param config
     *            ColumnConfig info
     * @param raw
     *            input column value
     * @param cutoff
     *            standard deviation cut off
     * @param isWeightedNorm
     *            if use weighted woe
     * @return normalized value for hybrid method.
     */
    private static List<Double> hybridNormalize(ColumnConfig config, String raw, Double cutoff, boolean isWeightedNorm) {
        List<Double> normValue;
        if(config.isNumerical()) {
            // For numerical data, use zscore.
            normValue = zScoreNormalize(config, raw, cutoff);
        } else {
            // For categorical data, use woe.
            normValue = woeNormalize(config, raw, isWeightedNorm);
        }

        return normValue;
    }

    /**
     * Check specified standard deviation cutoff and return the correct value.
     * 
     * @param cutoff
     *            specified standard deviation cutoff
     * @return If cutoff is valid then return it, else return {@link Normalizer#STD_DEV_CUTOFF}
     */
    public static double checkCutOff(Double cutoff) {
        double stdDevCutOff;
        if(cutoff != null && !cutoff.isInfinite() && !cutoff.isNaN()) {
            stdDevCutOff = cutoff;
        } else {
            stdDevCutOff = STD_DEV_CUTOFF;
        }

        return stdDevCutOff;
    }

    /**
     * Calculate woe mean and woe standard deviation.
     * 
     * @param config
     *            ColumnConfig info
     * @param isWeightedNorm
     *            if use weighted woe
     * @return an double array contains woe mean and woe standard deviation as order {mean, stdDev}
     */
    public static double[] calculateWoeMeanAndStdDev(ColumnConfig config, boolean isWeightedNorm) {
        List<Double> woeList = isWeightedNorm ? config.getBinWeightedWoe() : config.getBinCountWoe();
        if(woeList == null || woeList.size() < 2) {
            throw new IllegalArgumentException("Woe list is null or too short(size < 2)");
        }

        List<Integer> negCountList = config.getBinCountNeg();
        List<Integer> posCountList = config.getBinCountPos();

        // calculate woe mean and standard deviation
        int size = woeList.size();
        double sum = 0.0;
        double squaredSum = 0.0;
        long totalCount = 0;
        for(int i = 0; i < size; i++) {
            int count = negCountList.get(i) + posCountList.get(i);
            totalCount += count;
            double x = woeList.get(i);
            sum += x * count;
            squaredSum += x * x * count;
        }

        double woeMean = sum / totalCount;
        double woeStdDev = Math.sqrt(Math.abs((squaredSum - (sum * sum) / totalCount) / (totalCount - 1)));

        return new double[] { woeMean, woeStdDev };
    }

    /**
     * Compute the zscore, by original value, mean, standard deviation and standard deviation cutoff
     * 
     * @param var
     *            original value
     * @param mean
     *            mean value
     * @param stdDev
     *            standard deviation
     * @param stdDevCutOff
     *            standard deviation cutoff
     * @return zscore
     */
    public static Double[] computeZScore(double var, double mean, double stdDev, double stdDevCutOff) {
        double maxCutOff = mean + stdDevCutOff * stdDev;
        if(var > maxCutOff) {
            var = maxCutOff;
        }

        double minCutOff = mean - stdDevCutOff * stdDev;
        if(var < minCutOff) {
            var = minCutOff;
        }

        if(stdDev > 0.00001) {
            return new Double[] { (var - mean) / stdDev };
        } else {
            return new Double[] { 0.0 };
        }
    }

}
