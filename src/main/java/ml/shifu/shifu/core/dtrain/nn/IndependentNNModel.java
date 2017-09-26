/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.nn;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.GZIPInputStream;

import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.encog.ml.data.basic.BasicMLData;

/**
 * {@link IndependentNNModel} is a light NN engine to predict NN model, the only dependency is shifu, guagua and
 * encog-core jars.
 * 
 * <p>
 * Model format for IndependentNNModel is binary format and can only be read by this engine. With this engine, better
 * execution SLA is expected while model is not good to be read and modified.
 * 
 * {@link #loadFromStream(InputStream, boolean)} and {@link #loadFromStream(InputStream)} are the only two entry points
 * to instance a {@link IndependentNNModel}.
 * 
 * <p>
 * {@link #compute(Map)} are the two APIs called for prediction.
 * 
 * <p>
 * SLA is expected and tested better compared with PMML NN model.
 */
public class IndependentNNModel {

    /**
     * Encog based neural network instance which is used to compute nn score
     */
    private List<BasicFloatNetwork> basicNetworks;

    /**
     * Normalization type
     */
    private NormType normType;

    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private Map<Integer, String> numNameMap;

    /**
     * Mapping for (ColumnNum, Category List) for categorical feature
     */
    private Map<Integer, List<String>> cateCateMap;

    /**
     * Mapping for (ColumnNum, Category List) for categorical feature
     */
    private Map<Integer, ColumnType> columnTypeMap;

    /**
     * Mapping for (ColumnNum, index in double[] array), this is important to make input map with the input array be
     * consistent
     */
    private Map<Integer, Integer> columnNumIndexMap;

    /**
     * Mapping for (columnNum, (category, woeValue)) for categorical columns
     */
    private Map<Integer, Map<String, Double>> cateWoeMap;

    /**
     * Mapping for (columnNum, (category, weightedWoeValue)) for categorical columns
     */
    private Map<Integer, Map<String, Double>> cateWgtWoeMap;

    /**
     * Mapping for (columnNum, (category, posRate)) for categorical columns
     */
    private Map<Integer, Map<String, Double>> binPosRateMap;

    /**
     * Mapping for (columnNum, binBoundaries) for numerical columns
     */
    private Map<Integer, List<Double>> numerBinBoundaries;

    /**
     * Mapping for (columnNum, woes) for numerical columns; for hybrid, woe bins for both numerical and categorical
     * bins; last one in weightedWoes is for missing value bin
     */
    private Map<Integer, List<Double>> numerWoes;

    /**
     * Mapping for (columnNum, wgtWoes) for numerical columns; for hybrid, woe bins for both numerical and categorical
     * bins; last one in weightedBinWoes is for missing value bin
     */
    private Map<Integer, List<Double>> numerWgtWoes;

    /**
     * ZScore stddev cutoff value per each column
     */
    private Map<Integer, Double> cutOffMap;

    /**
     * Mapping for (columnNum, mean) for all columns
     */
    private Map<Integer, Double> numerMeanMap;

    /**
     * Mapping for (columnNum, stddev) for all columns
     */
    private Map<Integer, Double> numerStddevMap;

    /**
     * Mapping for (columnNum, woeMean) for all columns
     */
    private Map<Integer, Double> woeMeanMap;

    /**
     * Mapping for (columnNum, woeStddev) for all columns
     */
    private Map<Integer, Double> woeStddevMap;

    /**
     * Mapping for (columnNum, weightedWoeMean) for all columns
     */
    private Map<Integer, Double> wgtWoeMeanMap;

    /**
     * Mapping for (columnNum, weightedWoeStddev) for all columns
     */
    private Map<Integer, Double> wgtWoeStddevMap;

    /**
     * Model version
     */
    private static int version = CommonConstants.NN_FORMAT_VERSION;

    /**
     * Set it to private, {@link IndependentNNModel} can only be loaded by {@link #loadFromStream(InputStream)} and
     * {@link #loadFromStream(InputStream, boolean)}.
     */
    private IndependentNNModel(List<BasicFloatNetwork> basicNetworks, NormType normType,
            Map<Integer, String> numNameMappings, Map<Integer, List<String>> cateColumnNameNames,
            Map<Integer, Integer> columnNumIndexMap, Map<Integer, Map<String, Double>> cateWoeMap,
            Map<Integer, Map<String, Double>> wgtCateWoeMap, Map<Integer, Map<String, Double>> binPosRateMap,
            Map<Integer, List<Double>> numerBinBoundaries, Map<Integer, List<Double>> numerWgtWoes,
            Map<Integer, List<Double>> numerWoes, Map<Integer, Double> cutOffMap, Map<Integer, Double> numerMeanMap,
            Map<Integer, Double> numerStddevMap, Map<Integer, Double> woeMeanMap, Map<Integer, Double> woeStddevMap,
            Map<Integer, Double> wgtWoeMeanMap, Map<Integer, Double> wgtWoeStddevMap,
            Map<Integer, ColumnType> columnTypeMap) {
        this.basicNetworks = basicNetworks;
        this.normType = normType;
        this.numNameMap = numNameMappings;
        this.cateCateMap = cateColumnNameNames;
        this.columnNumIndexMap = columnNumIndexMap;
        this.cateWoeMap = cateWoeMap;
        this.cateWgtWoeMap = wgtCateWoeMap;
        this.binPosRateMap = binPosRateMap;
        this.numerBinBoundaries = numerBinBoundaries;
        this.numerWgtWoes = numerWgtWoes;
        this.numerWoes = numerWoes;
        this.cutOffMap = cutOffMap;
        this.numerMeanMap = numerMeanMap;
        this.numerStddevMap = numerStddevMap;
        this.woeMeanMap = woeMeanMap;
        this.woeStddevMap = woeStddevMap;
        this.wgtWoeMeanMap = wgtWoeMeanMap;
        this.wgtWoeStddevMap = wgtWoeStddevMap;
        this.columnTypeMap = columnTypeMap;
    }

    /**
     * Given double array data, compute score values of neural network
     * 
     * @param data
     *            data array includes only effective column data, numeric value is real value after normalization,
     *            categorical feature value is pos rates or woe .
     * @return neural network model output, if multiple models, do averaging on all models outputs
     */
    public double[] compute(double[] data) {
        if(this.basicNetworks == null || this.basicNetworks.size() == 0) {
            throw new IllegalStateException("no models inside");
        }

        if(this.basicNetworks.size() == 1) {
            return this.basicNetworks.get(0).compute(new BasicMLData(data)).getData();
        } else {
            int outputSize = this.basicNetworks.get(0).getOutputCount();
            int modelSize = this.basicNetworks.size();
            double[] results = new double[outputSize];
            for(BasicFloatNetwork network: this.basicNetworks) {
                double[] currResults = network.compute(new BasicMLData(data)).getData();
                assert currResults.length == results.length;
                for(int i = 0; i < currResults.length; i++) {
                    // directly do averaging on each model output element
                    results[i] += currResults[i] / modelSize;
                }
            }
            return results;
        }
    }

    /**
     * Given {@code dataMap} with format (columnName, value), compute score values of neural network model.
     * 
     * <p>
     * No any alert or exception if your {@code dataMap} doesn't contain features included in the model, such case will
     * be treated as missing value case. Please make sure feature names in keys of {@code dataMap} are consistent with
     * names in model.
     * 
     * <p>
     * In {@code dataMap}, numerical value can be (String, Double) format or (String, String) format, they will all be
     * parsed to Double; categorical value are all converted to (String, String). If value not in our categorical list,
     * it will also be treated as missing value.
     * 
     * <p>
     * In {@code dataMap}, data should be raw value and normalization is computed inside according to {@link #normType}
     * and stats information in such model.
     * 
     * @param dataMap
     *            {@code dataMap} for (columnName, value), numeric value can be double/String, categorical feature can
     *            be int(index) or category value. if not set or set to null, such feature will be treated as missing
     *            value. For numerical value, if it cannot be parsed successfully, it will also be treated as missing.
     * @return score output for neural network
     */
    public double[] compute(Map<String, Object> dataMap) {
        return compute(convertDataMapToDoubleArray(dataMap));
    }

    private double[] convertDataMapToDoubleArray(Map<String, Object> dataMap) {
        double[] data = new double[this.columnNumIndexMap.size()];
        for(Entry<Integer, Integer> entry: this.columnNumIndexMap.entrySet()) {
            double value = 0d;
            Integer columnNum = entry.getKey();
            String columnName = this.numNameMap.get(columnNum);
            Object obj = dataMap.get(columnName);
            ColumnType columnType = this.columnTypeMap.get(columnNum);
            if(columnType == ColumnType.C) {
                // categorical column
                switch(this.normType) {
                    case WOE:
                    case HYBRID:
                        value = getCategoricalWoeValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE:
                    case WEIGHT_HYBRID:
                        value = getCategoricalWoeValue(columnNum, obj, true);
                        break;
                    case WOE_ZSCORE:
                    case WOE_ZSCALE:
                        value = getCategoricalWoeZScoreValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE_ZSCORE:
                    case WEIGHT_WOE_ZSCALE:
                        value = getCategoricalWoeZScoreValue(columnNum, obj, true);
                        break;
                    case OLD_ZSCALE:
                    case OLD_ZSCORE:
                    case ZSCALE:
                    case ZSCORE:
                    default:
                        value = getCategoricalPosRateZScoreValue(columnNum, obj);
                        break;
                }
            } else if(columnType == ColumnType.N) {
                // numerical column
                switch(this.normType) {
                    case WOE:
                        value = getNumericalWoeValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE:
                        value = getNumericalWoeValue(columnNum, obj, true);
                        break;
                    case WOE_ZSCORE:
                    case WOE_ZSCALE:
                        value = getNumericalWoeZScoreValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE_ZSCORE:
                    case WEIGHT_WOE_ZSCALE:
                        value = getNumericalWoeZScoreValue(columnNum, obj, true);
                        break;
                    case OLD_ZSCALE:
                    case OLD_ZSCORE:
                    case ZSCALE:
                    case ZSCORE:
                    case HYBRID:
                    case WEIGHT_HYBRID:
                    default:
                        value = getNumericalZScoreValue(columnNum, obj);
                        break;
                }
            } else if(columnType == ColumnType.H) {
                // hybrid column
                switch(this.normType) {
                    case WOE:
                        value = getHybridWoeValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE:
                        value = getHybridWoeValue(columnNum, obj, true);
                        break;
                    case WOE_ZSCORE:
                    case WOE_ZSCALE:
                        value = getHybridWoeZScoreValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE_ZSCORE:
                    case WEIGHT_WOE_ZSCALE:
                        value = getHybridWoeZScoreValue(columnNum, obj, true);
                        break;
                    case OLD_ZSCALE:
                    case OLD_ZSCORE:
                    case ZSCALE:
                    case ZSCORE:
                    case HYBRID:
                    case WEIGHT_HYBRID:
                    default:
                        throw new IllegalStateException("Column type of " + columnName
                                + " is hybrid, but normType is not woe related.");
                }
            }

            Integer index = entry.getValue();
            if(index != null && index < data.length) {
                data[index] = value;
            }
        }
        return data;
    }

    private double getHybridWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        // for hybrid categories, category bin merge is not supported, so we can use
        List<String> binCategories = this.cateCateMap.get(columnNum);
        int binIndex = CommonUtils.getCategoicalBinIndex(binCategories, obj == null ? null : obj.toString());
        List<Double> binBoundaries = this.numerBinBoundaries.get(columnNum);
        if(binIndex != -1) {
            binIndex = binIndex + binBoundaries.size(); // append the first numerical bins
        } else {
            double douVal = CommonUtils.parseNumber(obj == null ? null : obj.toString());
            if(Double.isNaN(douVal)) {
                binIndex = binBoundaries.size() + binCategories.size();
            } else {
                binIndex = CommonUtils.getBinIndex(binBoundaries, douVal);
            }
        }

        List<Double> binWoes = isWeighted ? this.numerWgtWoes.get(columnNum) : this.numerWoes.get(columnNum);
        double value = 0d;
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value;
    }

    private double getNumericalWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        int binIndex = -1;
        if(obj != null) {
            binIndex = CommonUtils.getNumericalBinIndex(this.numerBinBoundaries.get(columnNum), obj.toString());
        }
        List<Double> binWoes = isWeighted ? this.numerWgtWoes.get(columnNum) : this.numerWoes.get(columnNum);

        double value = 0d;
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value;
    }

    private double getNumericalZScoreValue(Integer columnNum, Object obj) {
        double mean = this.numerMeanMap.get(columnNum);
        double stddev = this.numerStddevMap.get(columnNum);
        double rawValue = 0d;
        if(obj == null || obj.toString().length() == 0) {
            rawValue = defaultMissingValue(mean);
        } else {
            try {
                rawValue = Double.parseDouble(obj.toString());
            } catch (Exception e) {
                rawValue = defaultMissingValue(mean);
            }
        }
        double cutoff = Normalizer.checkCutOff(this.cutOffMap.get(columnNum));
        return Normalizer.computeZScore(rawValue, mean, stddev, cutoff)[0];
    }

    private double getCategoricalPosRateZScoreValue(Integer columnNum, Object obj) {
        double value = 0d;
        Map<String, Double> posRateMapping = this.binPosRateMap.get(columnNum);
        if(obj == null) {
            value = posRateMapping.get(Constants.EMPTY_CATEGORY);
        } else {
            Double posRate = posRateMapping.get(obj.toString());
            if(posRate == null) {
                value = posRateMapping.get(Constants.EMPTY_CATEGORY);
            } else {
                value = posRate;
            }
        }
        double mean = this.numerMeanMap.get(columnNum);
        double stddev = this.numerStddevMap.get(columnNum);
        double cutoff = Normalizer.checkCutOff(this.cutOffMap.get(columnNum));
        return Normalizer.computeZScore(value, mean, stddev, cutoff)[0];
    }

    private double getHybridWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getHybridWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMap : this.woeMeanMap;
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMap : this.woeStddevMap;
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double cutoff = Normalizer.checkCutOff(this.cutOffMap.get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, cutoff)[0];
    }

    private double getNumericalWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getNumericalWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMap : this.woeMeanMap;
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMap : this.woeStddevMap;
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double cutoff = Normalizer.checkCutOff(this.cutOffMap.get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, cutoff)[0];
    }

    private double getCategoricalWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getCategoricalWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMap : this.woeMeanMap;
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMap : this.woeStddevMap;
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double cutoff = Normalizer.checkCutOff(cutOffMap.get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, cutoff)[0];
    }

    private double getCategoricalWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        double value = 0d;
        Map<Integer, Map<String, Double>> mappings = isWeighted ? this.cateWgtWoeMap : cateWoeMap;
        Map<String, Double> woeMap = mappings.get(columnNum);
        if(obj == null) {
            value = woeMap.get(Constants.EMPTY_CATEGORY);
        } else {
            Double woe = woeMap.get(obj.toString());
            if(woe == null) {
                value = woeMap.get(Constants.EMPTY_CATEGORY);
            } else {
                value = woe;
            }
        }
        return value;
    }

    public static double defaultMissingValue(Double mean) {
        return mean == null ? 0 : mean.doubleValue();
    }

    /**
     * Load model instance from input stream which is saved in NNOutput for specified binary format.
     * 
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @return the nn model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */
    public static IndependentNNModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, true);
    }

    /**
     * Load model instance from input stream which is saved in NNOutput for specified binary format.
     * 
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @param isRemoveNameSpace
     *            is remove name space or not
     * @return the nn model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */
    public static IndependentNNModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        DataInputStream dis = null;
        // check if gzip or not
        try {
            byte[] header = new byte[2];
            BufferedInputStream bis = new BufferedInputStream(input);
            bis.mark(2);
            int result = bis.read(header);
            bis.reset();
            int ss = (header[0] & 0xff) | ((header[1] & 0xff) << 8);
            if(result != -1 && ss == GZIPInputStream.GZIP_MAGIC) {
                dis = new DataInputStream(new GZIPInputStream(bis));
            } else {
                dis = new DataInputStream(bis);
            }
        } catch (java.io.IOException e) {
            dis = new DataInputStream(input);
        }

        int version = dis.readInt();
        IndependentNNModel.setVersion(version);
        String normStr = ml.shifu.shifu.core.dtrain.StringUtils.readString(dis);
        NormType normType = NormType.valueOf(normStr.toUpperCase());

        // for all features
        Map<Integer, String> numNameMap = new HashMap<Integer, String>();
        Map<Integer, List<String>> cateColumnNameNames = new HashMap<Integer, List<String>>();
        // for categorical features
        Map<Integer, Map<String, Double>> cateWoeMap = new HashMap<Integer, Map<String, Double>>();
        Map<Integer, Map<String, Double>> cateWgtWoeMap = new HashMap<Integer, Map<String, Double>>();
        Map<Integer, Map<String, Double>> binPosRateMap = new HashMap<Integer, Map<String, Double>>();
        // for numerical features
        Map<Integer, List<Double>> numerBinBoundaries = new HashMap<Integer, List<Double>>();
        Map<Integer, List<Double>> numerWoes = new HashMap<Integer, List<Double>>();
        Map<Integer, List<Double>> numerWgtWoes = new HashMap<Integer, List<Double>>();
        // for all features
        Map<Integer, Double> numerMeanMap = new HashMap<Integer, Double>();
        Map<Integer, Double> numerStddevMap = new HashMap<Integer, Double>();
        Map<Integer, Double> woeMeanMap = new HashMap<Integer, Double>();
        Map<Integer, Double> woeStddevMap = new HashMap<Integer, Double>();
        Map<Integer, Double> wgtWoeMeanMap = new HashMap<Integer, Double>();
        Map<Integer, Double> wgtWoeStddevMap = new HashMap<Integer, Double>();
        Map<Integer, Double> cutoffMap = new HashMap<Integer, Double>();
        Map<Integer, ColumnType> columnTypeMap = new HashMap<Integer, ColumnType>();

        int columnSize = dis.readInt();
        for(int i = 0; i < columnSize; i++) {
            NNColumnStats cs = new NNColumnStats();
            cs.readFields(dis);

            List<Double> binWoes = cs.getBinCountWoes();
            List<Double> binWgtWoes = cs.getBinWeightWoes();
            List<Double> binPosRates = cs.getBinPosRates();

            int columnNum = cs.getColumnNum();

            columnTypeMap.put(columnNum, cs.getColumnType());

            if(isRemoveNameSpace) {
                // remove name-space in column name to make it be called by simple name
                numNameMap.put(columnNum, StringUtils.getSimpleColumnName(cs.getColumnName()));
            } else {
                numNameMap.put(columnNum, cs.getColumnName());
            }

            // for categorical features
            Map<String, Double> woeMap = new HashMap<String, Double>();
            Map<String, Double> woeWgtMap = new HashMap<String, Double>();
            Map<String, Double> posRateMap = new HashMap<String, Double>();

            if(cs.isCategorical() || cs.isHybrid()) {
                List<String> binCategories = cs.getBinCategories();

                cateColumnNameNames.put(columnNum, binCategories);
                for(int j = 0; j < binCategories.size(); j++) {
                    String currCate = binCategories.get(j);
                    if(currCate.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                        // merged category should be flatten, use own split function to avoid depending on guava jar in
                        // prediction
                        String[] splits = StringUtils.split(currCate, Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
                        for(String str: splits) {
                            woeMap.put(str, binWoes.get(j));
                            woeWgtMap.put(str, binWgtWoes.get(j));
                            posRateMap.put(str, binPosRates.get(j));
                        }
                    } else {
                        woeMap.put(currCate, binWoes.get(j));
                        woeWgtMap.put(currCate, binWgtWoes.get(j));
                        posRateMap.put(currCate, binPosRates.get(j));
                    }
                }
                // append last missing bin
                woeMap.put(Constants.EMPTY_CATEGORY, binWoes.get(binCategories.size()));
                woeWgtMap.put(Constants.EMPTY_CATEGORY, binWgtWoes.get(binCategories.size()));
                posRateMap.put(Constants.EMPTY_CATEGORY, binPosRates.get(binCategories.size()));
            }

            if(cs.isNumerical() || cs.isHybrid()) {
                numerBinBoundaries.put(columnNum, cs.getBinBoundaries());
                numerWoes.put(columnNum, binWoes);
                numerWgtWoes.put(columnNum, binWgtWoes);
            }

            cateWoeMap.put(columnNum, woeMap);
            cateWgtWoeMap.put(columnNum, woeWgtMap);
            binPosRateMap.put(columnNum, posRateMap);

            numerMeanMap.put(columnNum, cs.getMean());
            numerStddevMap.put(columnNum, cs.getStddev());
            woeMeanMap.put(columnNum, cs.getWoeMean());
            woeStddevMap.put(columnNum, cs.getWoeStddev());
            wgtWoeMeanMap.put(columnNum, cs.getWoeWgtMean());
            wgtWoeStddevMap.put(columnNum, cs.getWoeWgtStddev());
            cutoffMap.put(columnNum, cs.getCutoff());
        }

        Map<Integer, Integer> columnMap = new HashMap<Integer, Integer>();
        int columnMapSize = dis.readInt();
        for(int i = 0; i < columnMapSize; i++) {
            columnMap.put(dis.readInt(), dis.readInt());
        }

        int size = dis.readInt();
        List<BasicFloatNetwork> networks = new ArrayList<BasicFloatNetwork>();
        for(int i = 0; i < size; i++) {
            networks.add(new PersistBasicFloatNetwork().readNetwork(dis));
        }

        return new IndependentNNModel(networks, normType, numNameMap, cateColumnNameNames, columnMap, cateWoeMap,
                cateWgtWoeMap, binPosRateMap, numerBinBoundaries, numerWgtWoes, numerWoes, cutoffMap, numerMeanMap,
                numerStddevMap, woeMeanMap, woeStddevMap, wgtWoeMeanMap, wgtWoeStddevMap, columnTypeMap);
    }

    @SuppressWarnings("unused")
    private static DataInputStream ensureGzipIfExists(InputStream input) {
        // check if gzip or not
        DataInputStream dis = null;
        try {
            byte[] header = new byte[2];
            BufferedInputStream bis = new BufferedInputStream(input);
            bis.mark(2);
            int result = bis.read(header);
            bis.reset();
            int ss = (header[0] & 0xff) | ((header[1] & 0xff) << 8);
            if(result != -1 && ss == GZIPInputStream.GZIP_MAGIC) {
                dis = new DataInputStream(new GZIPInputStream(bis));
            } else {
                dis = new DataInputStream(bis);
            }
        } catch (java.io.IOException e) {
            dis = new DataInputStream(input);
        }
        return dis;
    }

    /**
     * @return the version
     */
    public static int getVersion() {
        return version;
    }

    /**
     * @param version
     *            the version to set
     */
    public static void setVersion(int version) {
        IndependentNNModel.version = version;
    }

    /**
     * @return the basicNetwork
     */
    public List<BasicFloatNetwork> getBasicNetworks() {
        return basicNetworks;
    }

    /**
     * @param basicNetworks
     *            the basicNetwork to set
     */
    public void setBasicNetwork(List<BasicFloatNetwork> basicNetworks) {
        this.basicNetworks = basicNetworks;
    }

    /**
     * @return the normType
     */
    public NormType getNormType() {
        return normType;
    }

    /**
     * @param normType
     *            the normType to set
     */
    public void setNormType(NormType normType) {
        this.normType = normType;
    }

    /**
     * @return the numNameMappings
     */
    public Map<Integer, String> getNumNameMappings() {
        return numNameMap;
    }

    /**
     * @param numNameMappings
     *            the numNameMappings to set
     */
    public void setNumNameMappings(Map<Integer, String> numNameMappings) {
        this.numNameMap = numNameMappings;
    }

    /**
     * @return the cateColumnNameNames
     */
    public Map<Integer, List<String>> getCateColumnNameNames() {
        return cateCateMap;
    }

    /**
     * @param cateColumnNameNames
     *            the cateColumnNameNames to set
     */
    public void setCateColumnNameNames(Map<Integer, List<String>> cateColumnNameNames) {
        this.cateCateMap = cateColumnNameNames;
    }

    /**
     * @return the columnNumIndexMap
     */
    public Map<Integer, Integer> getColumnNumIndexMap() {
        return columnNumIndexMap;
    }

    /**
     * @param columnNumIndexMap
     *            the columnNumIndexMap to set
     */
    public void setColumnNumIndexMap(Map<Integer, Integer> columnNumIndexMap) {
        this.columnNumIndexMap = columnNumIndexMap;
    }

    /**
     * @return the cateWoeMap
     */
    public Map<Integer, Map<String, Double>> getCateWoeMap() {
        return cateWoeMap;
    }

    /**
     * @param cateWoeMap
     *            the cateWoeMap to set
     */
    public void setCateWoeMap(Map<Integer, Map<String, Double>> cateWoeMap) {
        this.cateWoeMap = cateWoeMap;
    }

    /**
     * @return the wgtCateWoeMap
     */
    public Map<Integer, Map<String, Double>> getWgtCateWoeMap() {
        return cateWgtWoeMap;
    }

    /**
     * @param wgtCateWoeMap
     *            the wgtCateWoeMap to set
     */
    public void setWgtCateWoeMap(Map<Integer, Map<String, Double>> wgtCateWoeMap) {
        this.cateWgtWoeMap = wgtCateWoeMap;
    }

    /**
     * @return the binPosRateMap
     */
    public Map<Integer, Map<String, Double>> getBinPosRateMap() {
        return binPosRateMap;
    }

    /**
     * @param binPosRateMap
     *            the binPosRateMap to set
     */
    public void setBinPosRateMap(Map<Integer, Map<String, Double>> binPosRateMap) {
        this.binPosRateMap = binPosRateMap;
    }

    /**
     * @return the numerBinBoundaries
     */
    public Map<Integer, List<Double>> getNumerBinBoundaries() {
        return numerBinBoundaries;
    }

    /**
     * @param numerBinBoundaries
     *            the numerBinBoundaries to set
     */
    public void setNumerBinBoundaries(Map<Integer, List<Double>> numerBinBoundaries) {
        this.numerBinBoundaries = numerBinBoundaries;
    }

    /**
     * @return the numerWgtWoes
     */
    public Map<Integer, List<Double>> getNumerWgtWoes() {
        return numerWgtWoes;
    }

    /**
     * @param numerWgtWoes
     *            the numerWgtWoes to set
     */
    public void setNumerWgtWoes(Map<Integer, List<Double>> numerWgtWoes) {
        this.numerWgtWoes = numerWgtWoes;
    }

    /**
     * @return the numerWoes
     */
    public Map<Integer, List<Double>> getNumerWoes() {
        return numerWoes;
    }

    /**
     * @param numerWoes
     *            the numerWoes to set
     */
    public void setNumerWoes(Map<Integer, List<Double>> numerWoes) {
        this.numerWoes = numerWoes;
    }

    /**
     * @return the numerMeanMap
     */
    public Map<Integer, Double> getNumerMeanMap() {
        return numerMeanMap;
    }

    /**
     * @param numerMeanMap
     *            the numerMeanMap to set
     */
    public void setNumerMeanMap(Map<Integer, Double> numerMeanMap) {
        this.numerMeanMap = numerMeanMap;
    }

    /**
     * @return the numerStddevMap
     */
    public Map<Integer, Double> getNumerStddevMap() {
        return numerStddevMap;
    }

    /**
     * @param numerStddevMap
     *            the numerStddevMap to set
     */
    public void setNumerStddevMap(Map<Integer, Double> numerStddevMap) {
        this.numerStddevMap = numerStddevMap;
    }

    /**
     * @return the woeMeanMap
     */
    public Map<Integer, Double> getWoeMeanMap() {
        return woeMeanMap;
    }

    /**
     * @param woeMeanMap
     *            the woeMeanMap to set
     */
    public void setWoeMeanMap(Map<Integer, Double> woeMeanMap) {
        this.woeMeanMap = woeMeanMap;
    }

    /**
     * @return the woeStddevMap
     */
    public Map<Integer, Double> getWoeStddevMap() {
        return woeStddevMap;
    }

    /**
     * @param woeStddevMap
     *            the woeStddevMap to set
     */
    public void setWoeStddevMap(Map<Integer, Double> woeStddevMap) {
        this.woeStddevMap = woeStddevMap;
    }

    /**
     * @return the wgtWoeMeanMap
     */
    public Map<Integer, Double> getWgtWoeMeanMap() {
        return wgtWoeMeanMap;
    }

    /**
     * @param wgtWoeMeanMap
     *            the wgtWoeMeanMap to set
     */
    public void setWgtWoeMeanMap(Map<Integer, Double> wgtWoeMeanMap) {
        this.wgtWoeMeanMap = wgtWoeMeanMap;
    }

    /**
     * @return the wgtWoeStddevMap
     */
    public Map<Integer, Double> getWgtWoeStddevMap() {
        return wgtWoeStddevMap;
    }

    /**
     * @param wgtWoeStddevMap
     *            the wgtWoeStddevMap to set
     */
    public void setWgtWoeStddevMap(Map<Integer, Double> wgtWoeStddevMap) {
        this.wgtWoeStddevMap = wgtWoeStddevMap;
    }

    /**
     * @return the cutOffMap
     */
    public Map<Integer, Double> getCutOffMap() {
        return cutOffMap;
    }

    /**
     * @param cutOffMap
     *            the cutOffMap to set
     */
    public void setCutOffMap(Map<Integer, Double> cutOffMap) {
        this.cutOffMap = cutOffMap;
    }
}
