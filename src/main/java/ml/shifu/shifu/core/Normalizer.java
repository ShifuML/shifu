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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.udf.norm.CategoryMissingNormType;
import ml.shifu.shifu.util.BinUtils;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Util normalization class which is used for any kind of transformation.
 */
public class Normalizer {

    private static Logger log = LoggerFactory.getLogger(Normalizer.class);
    public static final double STD_DEV_CUTOFF = 4.0d;

    public enum NormalizeMethod {
        /**
         * Normalize methods.
         */
        ZScore, MaxMin
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
    public List<Double> normalize(Object raw) {
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
    public static List<Double> normalize(ColumnConfig config, Object raw) {
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
    public static List<Double> normalize(ColumnConfig config, Object raw, NormalizeMethod method) {
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
    public static List<Double> normalize(ColumnConfig config, Object raw, double stdDevCutoff) {
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
    public static List<Double> normalize(ColumnConfig config, Object raw, NormalizeMethod method, double stdDevCutoff) {
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
    private static Double[] getMaxMinScore(ColumnConfig config, Object raw) {
        if(config.isCategorical()) {
            // TODO, doesn't support
        } else {
            Double value = null;
            if(raw instanceof Double) {
                value = (Double) raw;
            } else if(raw instanceof Integer) {
                value = ((Integer) raw).doubleValue();
            } else {
                value = Double.parseDouble((String) raw);
            }
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
    public static List<Double> normalize(ColumnConfig config, Object raw, Double cutoff,
            ModelNormalizeConf.NormType type, CategoryMissingNormType categoryMissingNormType) {
        switch(type) {
            case ASIS_WOE:
                return asIsNormalize(config, raw, true);
            case ASIS_PR:
                return asIsNormalize(config, raw, false);
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
            case ONEHOT:
                return OneHotNormalize(config, raw);
            case ZSCALE_ONEHOT:
                return zscaleOneHotNormalize(config, raw, cutoff, categoryMissingNormType);
            case ZSCALE_ORDINAL:
                return zscaleOrdinalNormalize(config, raw, cutoff, categoryMissingNormType);
            case DISCRETE_ZSCORE:
            case DISCRETE_ZSCALE:
                return discreteZScoreNormalize(config, raw, cutoff, categoryMissingNormType);
            case OLD_ZSCALE:
            case OLD_ZSCORE:
                return zScoreNormalize(config, raw, cutoff, categoryMissingNormType, true);
            case ZSCALE:
            case ZSCORE:
            default:
                return zScoreNormalize(config, raw, cutoff, categoryMissingNormType, false);
        }
    }

    /**
     * Adding new API with cateIndeMap parameter without change normalize API.
     * 
     * @param config
     *            the ColumnConfig
     * @param raw
     *            the raw input
     * @param cutoff
     *            the cutoff value
     * @param type
     *            normalize type
     * @param categoryMissingNormType
     *            the category missing normal type
     * @param cateIndexMap
     *            the cateIndexMap map from category to index
     * @return normalized value
     */
    public static List<Double> fullNormalize(ColumnConfig config, Object raw, Double cutoff,
            ModelNormalizeConf.NormType type, CategoryMissingNormType categoryMissingNormType,
            Map<String, Integer> cateIndexMap) {
        switch(type) {
            case ZSCORE_INDEX:
            case ZSCALE_INDEX:
                return numZScoreAndCateIndexNorm(config, raw, cutoff, cateIndexMap);
            case WOE_INDEX:
                if(config.isNumerical()) {
                    return woeNormalize(config, raw, false);
                } else if(config.isCategorical()) {
                    Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
                    if(index == null || index == -1) {
                        // last index for null category
                        index = config.getBinCategory().size();
                    }
                    return Arrays.asList((double) index);
                }
            case WOE_ZSCALE_INDEX:
                if(config.isNumerical()) {
                    return woeZScoreNormalize(config, raw, cutoff, false);
                } else if(config.isCategorical()) {
                    Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
                    if(index == null || index == -1) {
                        // last index for null category
                        index = config.getBinCategory().size();
                    }
                    return Arrays.asList((double) index);
                }
            case INDEX:
                if(config.isNumerical()) {
                    int binIndex = BinUtils.getBinNum(config, raw);
                    if(binIndex < 0 || binIndex > config.getBinBoundary().size()) {
                        binIndex = config.getBinBoundary().size();
                    }
                    return Arrays.asList((double) binIndex);
                } else if(config.isCategorical()) {
                    Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
                    if(index == null || index == -1) {
                        // last index for null category
                        index = config.getBinCategory().size();
                    }
                    return Arrays.asList((double) index);
                }
            case ZSCORE_APPEND_INDEX:
            case ZSCALE_APPEND_INDEX:
                List<Double> zscores = numZScoreAndCateIndexNorm(config, raw, cutoff, cateIndexMap);
                if(config.isNumerical()) {
                    int binIndex = BinUtils.getBinNum(config, raw);
                    if(binIndex < 0 || binIndex > config.getBinBoundary().size()) {
                        binIndex = config.getBinBoundary().size();
                    }
                    return Arrays.asList(zscores.get(0), (double) binIndex);
                } else if(config.isCategorical()) {
                    Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
                    if(index == null || index == -1) {
                        // last index for null category
                        index = config.getBinCategory().size();
                    }
                    return Arrays.asList(zscores.get(0), (double) index);
                }
            case WOE_APPEND_INDEX:
                List<Double> zWoeScores = woeNormalize(config, raw, false);
                if(config.isNumerical()) {
                    int binIndex = BinUtils.getBinNum(config, raw);
                    if(binIndex < 0 || binIndex > config.getBinBoundary().size()) {
                        binIndex = config.getBinBoundary().size();
                    }
                    return Arrays.asList(zWoeScores.get(0), (double) binIndex);
                } else if(config.isCategorical()) {
                    Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
                    if(index == null || index == -1) {
                        // last index for null category
                        index = config.getBinCategory().size();
                    }
                    return Arrays.asList(zWoeScores.get(0), (double) index);
                }
            case WOE_ZSCALE_APPEND_INDEX:
                List<Double> zWoeZScores = woeZScoreNormalize(config, raw, cutoff, false);
                if(config.isNumerical()) {
                    int binIndex = BinUtils.getBinNum(config, raw);
                    if(binIndex < 0 || binIndex > config.getBinBoundary().size()) {
                        binIndex = config.getBinBoundary().size();
                    }
                    return Arrays.asList(zWoeZScores.get(0), (double) binIndex);
                } else if(config.isCategorical()) {
                    Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
                    if(index == null || index == -1) {
                        // last index for null category
                        index = config.getBinCategory().size();
                    }
                    return Arrays.asList(zWoeZScores.get(0), (double) index);
                }
            default:
                // others use old normalize API to reuse code
                return normalize(config, raw, cutoff, type, categoryMissingNormType);
        }
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
     * @param cateIndexMap
     *            missing categorical value norm type
     * @return normalized value for ZScore method.
     */
    private static List<Double> numZScoreAndCateIndexNorm(ColumnConfig config, Object raw, Double cutoff,
            Map<String, Integer> cateIndexMap) {
        if(config.isNumerical()) {
            double stdDevCutOff = checkCutOff(cutoff);
            double value = parseRawValue(config, raw, null);
            return Arrays.asList(computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff));
        } else if(config.isCategorical()) {
            Integer index = cateIndexMap == null ? null : cateIndexMap.get(raw == null ? "" : raw.toString());
            if(index == null || index == -1) {
                // last index for null category
                index = config.getBinCategory().size();
            }
            return Arrays.asList(((double) index));
        } else {
            throw new IllegalArgumentException("Not supported norm column type.");
        }
    }

    private static List<Double> asIsNormalize(ColumnConfig config, Object raw, boolean toUseWoe) {
        if(config.isNumerical()) {
            Double values[] = new Double[1];
            if(raw instanceof Double) {
                values[0] = (Double) raw;
            } else if(raw instanceof Integer) {
                values[0] = ((Integer) raw).doubleValue();
            } else {
                try {
                    values[0] = Double.parseDouble(raw.toString());
                } catch (Exception e) {
                    log.warn("Illegal numerical value - {}, use mean instead.", raw);
                    values[0] = config.getMean();
                }
            }

            return Arrays.asList(values);
        } else {
            // categorical variables
            List<Double> normVals = (toUseWoe ? config.getBinCountWoe() : config.getBinPosRate());
            int binIndex = BinUtils.getBinNum(config, raw);
            return ((binIndex == -1) ? Arrays.asList(new Double[] { normVals.get(normVals.size() - 1) })
                    : Arrays.asList(new Double[] { normVals.get(binIndex) }));
        }
    }

    private static List<Double> OneHotNormalize(ColumnConfig config, Object raw) {
        Double[] normData = (config.isNumerical() ? new Double[config.getBinBoundary().size() + 1]
                : new Double[config.getBinCategory().size() + 1]);
        Arrays.fill(normData, 0.0d);
        int binNum = BinUtils.getBinNum(config, raw);
        if(binNum < 0) {
            binNum = normData.length - 1;
        }
        normData[binNum] = 1.0d;
        return Arrays.asList(normData);
    }

    private static List<Double> zscaleOrdinalNormalize(ColumnConfig config, Object raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType) {
        if(config.isNumerical()) {
            return zScoreNormalize(config, raw, cutoff, categoryMissingNormType, false);
        } else {
            int binNum = BinUtils.getBinNum(config, raw);
            if(binNum < 0) {
                binNum = config.getBinCategory().size();
            }
            Double[] normVals = new Double[] { (double) binNum };
            return Arrays.asList(normVals);
        }
    }

    private static List<Double> zscaleOneHotNormalize(ColumnConfig config, Object raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType) {
        if(config.isNumerical()) {
            return zScoreNormalize(config, raw, cutoff, categoryMissingNormType, false);
        } else {
            Double[] normData = new Double[config.getBinCategory().size() + 1];
            Arrays.fill(normData, 0.0d);

            int binNum = BinUtils.getBinNum(config, raw);
            if(binNum < 0) {
                binNum = config.getBinCategory().size();
            }
            normData[binNum] = 1.0d;
            return Arrays.asList(normData);
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
    public static List<Double> normalize(ColumnConfig config, Object raw, Double cutoff,
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
    private static List<Double> zScoreNormalize(ColumnConfig config, Object raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType, boolean isOld) {
        double stdDevCutOff = checkCutOff(cutoff);
        double value = parseRawValue(config, raw, categoryMissingNormType);
        if(isOld && config.isCategorical()) {
            return Arrays.asList(value);
        }
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
    private static List<Double> discreteZScoreNormalize(ColumnConfig config, Object raw, Double cutoff,
            CategoryMissingNormType categoryMissingNormType) {
        double stdDevCutOff = checkCutOff(cutoff);
        double value = 0;
        if(config.isCategorical()) {
            value = parseRawValue(config, raw, categoryMissingNormType);
        } else {
            int binIndex = BinUtils.getBinNum(config, raw);
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
    private static List<Double> zScoreNormalize(ColumnConfig config, Object raw, Double cutoff) {
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
    private static double parseRawValue(ColumnConfig config, Object raw,
            CategoryMissingNormType categoryMissingNormType) {
        if(categoryMissingNormType == null) {
            categoryMissingNormType = CategoryMissingNormType.POSRATE;
        }
        double value = 0.0;
        if(raw == null || StringUtils.isBlank(raw.toString())) {
            log.debug("Not decimal format but null, using default!");
            if(config.isCategorical()) {
                value = fillDefaultValue(config, categoryMissingNormType);
            } else {
                value = defaultMissingValue(config);
            }
            return value;
        }

        if(config.isCategorical()) {
            // for categorical variable, no need convert to double but double should be in treated as String in
            // categorical variables
            int index = BinUtils.getBinNum(config, raw);
            if(index == -1) {
                value = fillDefaultValue(config, categoryMissingNormType);
            } else {
                Double binPosRate = config.getBinPosRate().get(index);
                if(binPosRate != null) {
                    value = binPosRate.doubleValue();
                } else {
                    value = fillDefaultValue(config, categoryMissingNormType);
                }
            }
        } else {
            // for numerical value, if double or int, no need parse again.
            if(raw instanceof Double) {
                value = (Double) raw;
            } else if(raw instanceof Integer) {
                value = ((Integer) raw).doubleValue();
            } else if(raw instanceof Float) {
                value = ((Float) raw).doubleValue();
            } else {
                try {
                    // if raw is NaN, it won't throw Exception. The value will be Double.NaN
                    value = Double.parseDouble(raw.toString());
                } catch (Exception e) {
                    log.debug("Not decimal format " + raw + ", using default!");
                    value = defaultMissingValue(config);
                }
            }
            if(Double.isInfinite(value) || Double.isNaN(value)) {
                // if the value is Infinite or NaN, treat it as missing value
                // should treat Infinite as missing value?
                value = defaultMissingValue(config);
            }
        }

        return value;
    }

    private static double fillDefaultValue(ColumnConfig config, CategoryMissingNormType categoryMissingNormType) {
        double value = 0.0;
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
    private static List<Double> woeNormalize(ColumnConfig config, Object raw, boolean isWeightedNorm) {
        List<Double> woeBins = isWeightedNorm ? config.getBinWeightedWoe() : config.getBinCountWoe();
        int binIndex = 0;
        if(config.isHybrid()) {
            if(raw == null) {
                binIndex = -1;
            } else {
                binIndex = BinUtils.getCategoicalBinIndex(config, raw.toString());
            }

            if(binIndex != -1) {
                binIndex = binIndex + config.getBinBoundary().size(); // append the first numerical bins
            } else {
                double douVal = BinUtils.parseNumber(raw);
                if(Double.isNaN(douVal)) {
                    binIndex = config.getBinBoundary().size() + config.getBinCategory().size();
                } else {
                    binIndex = BinUtils.getBinIndex(config.getBinBoundary(), douVal);
                }
            }
        } else {
            binIndex = BinUtils.getBinNum(config, raw);
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
    private static List<Double> woeZScoreNormalize(ColumnConfig config, Object raw, Double cutoff,
            boolean isWeightedNorm) {
        double stdDevCutOff = checkCutOff(cutoff);
        double woe = woeNormalize(config, raw, isWeightedNorm).get(0);
        // TODO cache such computing to avoid computing each time
        double[] meanAndStdDev = calculateWoeMeanAndStdDev(config, isWeightedNorm);
        return Arrays.asList(computeZScore(woe, meanAndStdDev[0], meanAndStdDev[1], stdDevCutOff));
    }

    /**
     * Compute the normalized data for hybrid normalize. Use zscore noramlize for numerical data. Use woe normalize
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
    private static List<Double> hybridNormalize(ColumnConfig config, Object raw, Double cutoff,
            boolean isWeightedNorm) {
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
