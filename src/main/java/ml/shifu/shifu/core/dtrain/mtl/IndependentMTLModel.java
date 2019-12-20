/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.mtl;

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

import org.encog.mathutil.BoundMath;

import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.core.dtrain.nn.NNColumnStats;
import ml.shifu.shifu.util.BinUtils;
import ml.shifu.shifu.util.Constants;

/**
 * {@link IndependentMTLModel} is a light MTL engine to predict MTL model, the only dependency is shifu, guagua.
 * 
 * <p>
 * {@link #compute(Map)} is the main predict method from raw values. Normalization would be embedded and then do
 * inference to get a list of scores.
 * 
 * <p>
 * {@link #loadFromStream(InputStream)} and {@link #loadFromStream(InputStream, boolean)} are two utility methods to
 * load MTL binary model spec as a {@link IndependentMTLModel} instance.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class IndependentMTLModel {

    /**
     * Multi-task model graph definition network.
     */
    private MultiTaskModel mtm;

    /**
     * Normalization type
     */
    private NormType normType;

    /**
     * Model format version
     */
    private static int version = CommonConstants.MTL_FORMAT_VERSION;

    /**
     * Mapping for (ColumnNum, Map(Category, CategoryIndex) for categorical feature
     */
    private List<Map<Integer, Map<String, Integer>>> cateIndexMapList;
    /**
     * ZScore stddev cutoff value per each column
     */
    private List<Map<Integer, Double>> cutOffMapList;
    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private List<Map<Integer, String>> numNameMapList;
    /**
     * Mapping for (columnNum, binBoundaries) for numerical columns
     */
    private List<Map<Integer, List<Double>>> numerBinBoundaryMapList;
    /**
     * Mapping for (columnNum, woes) for numerical columns; for hybrid, woe bins for both numerical and categorical
     * bins; last one in weightedWoes is for missing value bin
     */
    private List<Map<Integer, List<Double>>> numerWoeMapList;
    /**
     * Mapping for (columnNum, wgtWoes) for numerical columns; for hybrid, woe bins for both numerical and categorical
     * bins; last one in weightedBinWoes is for missing value bin
     */
    private List<Map<Integer, List<Double>>> numerWgtWoeMapList;
    /**
     * Mapping for (columnNum, mean) for all columns
     */
    private List<Map<Integer, Double>> numerMeanMapList;
    /**
     * Mapping for (columnNum, stddev) for all columns
     */
    private List<Map<Integer, Double>> numerStddevMapList;
    /**
     * Mapping for (columnNum, woeMean) for all columns
     */
    private List<Map<Integer, Double>> woeMeanMapList;
    /**
     * Mapping for (columnNum, woeStddev) for all columns
     */
    private List<Map<Integer, Double>> woeStddevMapList;
    /**
     * Mapping for (columnNum, weightedWoeMean) for all columns
     */
    private List<Map<Integer, Double>> wgtWoeMeanMapList;
    /**
     * Mapping for (columnNum, weightedWoeStddev) for all columns
     */
    private List<Map<Integer, Double>> wgtWoeStddevMapList;
    /**
     * Mapping for (ColumnNum, index in double[] array)
     */
    private List<Map<Integer, Integer>> columnNumIndexMapList;

    private List<Map<Integer, ColumnType>> columnTypeMapList;

    private List<Map<Integer, Map<String, Double>>> binPosRateMapList = new ArrayList<>();

    private List<Map<Integer, List<String>>> cateColumnNameMapList;

    private List<Map<Integer, Map<String, Double>>> cateWoeMapList = new ArrayList<>();

    private List<Map<Integer, Map<String, Double>>> cateWgtWoeMapList = new ArrayList<>();

    private IndependentMTLModel(MultiTaskModel mtm, NormType normType, List<Map<Integer, Double>> cutOffMapList,
            List<Map<Integer, String>> numNameMapList, List<Map<Integer, Map<String, Integer>>> cateIndexMapList,
            List<Map<Integer, List<Double>>> numerBinBoundaryMapList, List<Map<Integer, List<Double>>> numerWoeMapList,
            List<Map<Integer, List<Double>>> numerWgtWoeMapList, List<Map<Integer, Double>> numerMeanMapList,
            List<Map<Integer, Double>> numerStddevMapList, List<Map<Integer, Double>> woeMeanMapList,
            List<Map<Integer, Double>> woeStddevMapList, List<Map<Integer, Double>> wgtWoeMeanMapList,
            List<Map<Integer, Double>> wgtWoeStddevMapList, List<Map<Integer, Integer>> columnNumIndexMapList,
            List<Map<Integer, ColumnType>> columnTypeMapList, List<Map<Integer, Map<String, Double>>> binPosRateMapList,
            List<Map<Integer, List<String>>> cateColumnNameMapList,
            List<Map<Integer, Map<String, Double>>> cateWoeMapList,
            List<Map<Integer, Map<String, Double>>> cateWgtWoeMapList) {
        this.mtm = mtm;
        this.normType = normType;
        this.cutOffMapList = cutOffMapList;
        this.numNameMapList = numNameMapList;
        this.cateIndexMapList = cateIndexMapList;
        this.numerBinBoundaryMapList = numerBinBoundaryMapList;
        this.numerWoeMapList = numerWoeMapList;
        this.numerWgtWoeMapList = numerWgtWoeMapList;
        this.numerMeanMapList = numerMeanMapList;
        this.numerStddevMapList = numerStddevMapList;
        this.woeMeanMapList = woeMeanMapList;
        this.woeStddevMapList = woeStddevMapList;
        this.wgtWoeMeanMapList = wgtWoeMeanMapList;
        this.wgtWoeStddevMapList = wgtWoeStddevMapList;
        this.columnNumIndexMapList = columnNumIndexMapList;
        this.columnTypeMapList = columnTypeMapList;
        this.binPosRateMapList = binPosRateMapList;
        this.cateColumnNameMapList = cateColumnNameMapList;
        this.cateWoeMapList = cateWoeMapList;
        this.cateWgtWoeMapList = cateWgtWoeMapList;
    }

    /**
     * Compute sigmoid scores according to data inputs after normalizations.
     *
     * @param inputs
     *            the dense inputs for deep model, numerical values
     * @return model score of the inputs.
     */
    public double[] compute(double[] inputs) {
        double[] logits = this.mtm.forward(inputs);
        assert logits != null && logits.length > 0;
        double[] scores = new double[logits.length];
        for(int i = 0; i < scores.length; i++) {
            scores[i] = sigmoid(logits[i]);
        }
        return scores;
    }

    /**
     * Sigmoid function. To have here which no need depends on CommonUtils which introduce more hadoop libs, not good
     * for production engine dependenct management.
     * 
     * @param logit
     *            the logit before sigmoid
     * @return sigmoid value
     */
    public double sigmoid(double logit) {
        return 1.0d / (1.0d + BoundMath.exp(-1 * logit));
    }

    /**
     * Given {@code dataMap} with format (columnName, value), compute score values of mtl model.
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
     *            {@code dataMap} for (columnName, value), numerical value can be double/String, categorical feature can
     *            be int(index) or category value. if not set or set to null, such feature will be treated as missing
     *            value. For numerical value, if it cannot be parsed successfully, it will also be treated as missing.
     * @return score outputs for MTLParams model
     */
    public double[] compute(Map<String, Object> dataMap) {
        return compute(convertDataMapToDoubleArray(dataMap));
    }

    /**
     * Load model instance from input stream which is saved in WDLOutput for specified binary format.
     *
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @return the nn model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */
    public static IndependentMTLModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, true);
    }

    /**
     * Load model instance from input stream which is saved in WDLOutput for specified binary format.
     *
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @param isRemoveNameSpace,
     *            is remove name space or not
     * @return the MTL model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */

    public static IndependentMTLModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        // check if gzip or not
        DataInputStream dis = checkGzipStream(input);

        int version = dis.readInt();
        IndependentMTLModel.setVersion(version);
        // Reserved two double field, one double field and one string field
        dis.readDouble();
        dis.readDouble();
        dis.readDouble();
        dis.readUTF();

        // read normStr
        String normStr = StringUtils.readString(dis);
        NormType normType = NormType.valueOf(normStr != null ? normStr.toUpperCase() : null);

        int mtlSize = dis.readInt();
        assert mtlSize > 0;
        List<Map<Integer, String>> numNameMapList = new ArrayList<>();
        List<Map<Integer, List<Double>>> numerBinBoundaryList = new ArrayList<>();
        List<Map<Integer, List<Double>>> numerWoeList = new ArrayList<>();
        List<Map<Integer, List<Double>>> numerWgtWoeList = new ArrayList<>();
        // for all features
        List<Map<Integer, Double>> numerMeanMapList = new ArrayList<>();
        List<Map<Integer, Double>> numerStddevMapList = new ArrayList<>();
        List<Map<Integer, Double>> woeMeanMapList = new ArrayList<>();
        List<Map<Integer, Double>> woeStddevMapList = new ArrayList<>();
        List<Map<Integer, Double>> wgtWoeMeanMapList = new ArrayList<>();
        List<Map<Integer, Double>> wgtWoeStddevMapList = new ArrayList<>();
        List<Map<Integer, Double>> cutoffMapList = new ArrayList<>();
        List<Map<Integer, Map<String, Integer>>> cateIndexMapList = new ArrayList<>();
        List<Map<Integer, Integer>> columnMapList = new ArrayList<>();
        List<Map<Integer, ColumnType>> columnTypeMapList = new ArrayList<>();
        List<Map<Integer, Map<String, Double>>> binPosRateMapList = new ArrayList<>();
        List<Map<Integer, List<String>>> cateColumnNameMapList = new ArrayList<>();
        List<Map<Integer, Map<String, Double>>> cateWoeMapList = new ArrayList<>();
        List<Map<Integer, Map<String, Double>>> cateWgtWoeMapList = new ArrayList<>();

        for(int i = 0; i < mtlSize; i++) {
            int columnSize = dis.readInt();
            // for all features
            Map<Integer, String> numNameMap = new HashMap<>(columnSize);
            // for numerical features
            Map<Integer, List<Double>> numerBinBoundaries = new HashMap<>(columnSize);
            Map<Integer, List<Double>> numerWoes = new HashMap<>(columnSize);
            Map<Integer, List<Double>> numerWgtWoes = new HashMap<>(columnSize);
            // for all features
            Map<Integer, Double> numerMeanMap = new HashMap<>(columnSize);
            Map<Integer, Double> numerStddevMap = new HashMap<>(columnSize);
            Map<Integer, Double> woeMeanMap = new HashMap<>(columnSize);
            Map<Integer, Double> woeStddevMap = new HashMap<>(columnSize);
            Map<Integer, Double> wgtWoeMeanMap = new HashMap<>(columnSize);
            Map<Integer, Double> wgtWoeStddevMap = new HashMap<>(columnSize);
            Map<Integer, Double> cutoffMap = new HashMap<>(columnSize);
            Map<Integer, ColumnType> columnTypeMap = new HashMap<Integer, ColumnType>();
            Map<Integer, Map<String, Integer>> cateIndexMapping = new HashMap<>(columnSize);
            Map<Integer, Map<String, Double>> binPosRateMap = new HashMap<Integer, Map<String, Double>>();
            Map<Integer, List<String>> cateColumnNameNames = new HashMap<Integer, List<String>>();
            Map<Integer, Map<String, Double>> cateWoeMap = new HashMap<Integer, Map<String, Double>>();
            Map<Integer, Map<String, Double>> cateWgtWoeMap = new HashMap<Integer, Map<String, Double>>();

            for(int j = 0; j < columnSize; j++) {
                NNColumnStats cs = new NNColumnStats();
                cs.readFields(dis);

                List<Double> binWoes = cs.getBinCountWoes();
                List<Double> binWgtWoes = cs.getBinWeightWoes();
                List<Double> binPosRates = cs.getBinPosRates();

                int columnNum = cs.getColumnNum();

                if(isRemoveNameSpace) {
                    // remove name-space in column name to make it be called by simple name
                    numNameMap.put(columnNum, StringUtils.getSimpleColumnName(cs.getColumnName()));
                } else {
                    numNameMap.put(columnNum, cs.getColumnName());
                }

                // for categorical features
                Map<String, Integer> cateIndexMap = new HashMap<>(cs.getBinCategories().size());
                Map<String, Double> posRateMap = new HashMap<String, Double>();
                Map<String, Double> woeMap = new HashMap<String, Double>();
                Map<String, Double> woeWgtMap = new HashMap<String, Double>();

                if(cs.isCategorical() || cs.isHybrid()) {
                    List<String> binCategories = cs.getBinCategories();

                    cateColumnNameNames.put(columnNum, binCategories);
                    for(int k = 0; k < binCategories.size(); k++) {
                        String currCate = binCategories.get(k);
                        if(currCate.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                            // merged category should be flatten, use own split function to avoid depending on guava jar
                            // in prediction
                            String[] splits = StringUtils.split(currCate, Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
                            for(String str: splits) {
                                woeMap.put(str, binWoes.get(k));
                                woeWgtMap.put(str, binWgtWoes.get(k));
                                posRateMap.put(str, binPosRates.get(k));
                                cateIndexMap.put(str, k);
                            }
                        } else {
                            woeMap.put(currCate, binWoes.get(k));
                            woeWgtMap.put(currCate, binWgtWoes.get(k));
                            posRateMap.put(currCate, binPosRates.get(k));
                            cateIndexMap.put(currCate, k);
                        }
                    }
                }

                if(cs.isNumerical() || cs.isHybrid()) {
                    numerBinBoundaries.put(columnNum, cs.getBinBoundaries());
                    numerWoes.put(columnNum, binWoes);
                    numerWgtWoes.put(columnNum, binWgtWoes);
                }

                cateIndexMapping.put(columnNum, cateIndexMap);
                numerMeanMap.put(columnNum, cs.getMean());
                numerStddevMap.put(columnNum, cs.getStddev());
                woeMeanMap.put(columnNum, cs.getWoeMean());
                woeStddevMap.put(columnNum, cs.getWoeStddev());
                wgtWoeMeanMap.put(columnNum, cs.getWoeWgtMean());
                wgtWoeStddevMap.put(columnNum, cs.getWoeWgtStddev());
                cutoffMap.put(columnNum, cs.getCutoff());
                columnTypeMap.put(columnNum, cs.getColumnType());
                binPosRateMap.put(columnNum, posRateMap);
                cateWoeMap.put(columnNum, woeMap);
                cateWgtWoeMap.put(columnNum, woeWgtMap);
            }

            int columnMappingSize = dis.readInt();
            Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnMappingSize, 1f);
            for(int k = 0; k < columnMappingSize; k++) {
                columnMapping.put(dis.readInt(), dis.readInt());
            }

            numNameMapList.add(numNameMap);
            numerBinBoundaryList.add(numerBinBoundaries);
            numerWoeList.add(numerWoes);
            numerWgtWoeList.add(numerWgtWoes);
            numerMeanMapList.add(numerMeanMap);
            numerStddevMapList.add(numerStddevMap);
            woeMeanMapList.add(woeMeanMap);
            woeStddevMapList.add(woeStddevMap);
            wgtWoeMeanMapList.add(wgtWoeMeanMap);
            wgtWoeStddevMapList.add(wgtWoeStddevMap);
            cutoffMapList.add(cutoffMap);
            cateIndexMapList.add(cateIndexMapping);
            columnMapList.add(columnMapping);
            binPosRateMapList.add(binPosRateMap);
            cateColumnNameMapList.add(cateColumnNameNames);
            cateWoeMapList.add(cateWoeMap);
            cateWgtWoeMapList.add(cateWgtWoeMap);
        }

        MultiTaskModel mtm = new MultiTaskModel();
        mtm.readFields(dis);
        return new IndependentMTLModel(mtm, normType, cutoffMapList, numNameMapList, cateIndexMapList,
                numerBinBoundaryList, numerWoeList, numerWgtWoeList, numerMeanMapList, numerStddevMapList,
                woeMeanMapList, woeStddevMapList, wgtWoeMeanMapList, wgtWoeStddevMapList, columnMapList,
                columnTypeMapList, binPosRateMapList, cateColumnNameMapList, cateWoeMapList, cateWgtWoeMapList);
    }

    private static DataInputStream checkGzipStream(InputStream input) {
        DataInputStream dis;
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
        return IndependentMTLModel.version;
    }

    /**
     * @param version
     *            the version to set
     */
    public static void setVersion(int version) {
        IndependentMTLModel.version = version;
    }

    private double[] convertDataMapToDoubleArray(Map<String, Object> dataMap) {
        int inputCounts = 0;
        for(int i = 0; i < this.columnNumIndexMapList.size(); i++) {
            inputCounts += this.columnNumIndexMapList.get(i).size();
        }

        double[] data = new double[inputCounts];
        int currSize = 0;
        for(int i = 0; i < this.columnNumIndexMapList.size(); i++) {
            if(i == 0) {
                currSize = 0;
            } else {
                currSize += this.columnNumIndexMapList.get(i - 1).size();
            }
            Map<Integer, Integer> colNumIndexMap = this.columnNumIndexMapList.get(i);

            for(Entry<Integer, Integer> entry: colNumIndexMap.entrySet()) {
                double value = 0d;
                Integer columnNum = entry.getKey();
                String columnName = this.numNameMapList.get(i).get(columnNum);
                Object obj = dataMap.get(columnName);
                ColumnType columnType = this.columnTypeMapList.get(i).get(columnNum);
                if(columnType == ColumnType.C) {
                    // categorical column
                    switch(this.normType) {
                        case WOE:
                        case HYBRID:
                            value = getCategoricalWoeValue(i, columnNum, obj, false);
                            break;
                        case WEIGHT_WOE:
                        case WEIGHT_HYBRID:
                            value = getCategoricalWoeValue(i, columnNum, obj, true);
                            break;
                        case WOE_ZSCORE:
                        case WOE_ZSCALE:
                            value = getCategoricalWoeZScoreValue(i, columnNum, obj, false);
                            break;
                        case WEIGHT_WOE_ZSCORE:
                        case WEIGHT_WOE_ZSCALE:
                            value = getCategoricalWoeZScoreValue(i, columnNum, obj, true);
                            break;
                        case OLD_ZSCALE:
                        case OLD_ZSCORE:
                            value = getCategoricalPosRateZScoreValue(i, columnNum, obj, true);
                            break;
                        case ZSCALE:
                        case ZSCORE:
                        default:
                            value = getCategoricalPosRateZScoreValue(i, columnNum, obj, false);
                            break;
                    }
                } else if(columnType == ColumnType.N) {
                    // numerical column
                    switch(this.normType) {
                        case WOE:
                            value = getNumericalWoeValue(i, columnNum, obj, false);
                            break;
                        case WEIGHT_WOE:
                            value = getNumericalWoeValue(i, columnNum, obj, true);
                            break;
                        case WOE_ZSCORE:
                        case WOE_ZSCALE:
                            value = getNumericalWoeZScoreValue(i, columnNum, obj, false);
                            break;
                        case WEIGHT_WOE_ZSCORE:
                        case WEIGHT_WOE_ZSCALE:
                            value = getNumericalWoeZScoreValue(i, columnNum, obj, true);
                            break;
                        case OLD_ZSCALE:
                        case OLD_ZSCORE:
                        case ZSCALE:
                        case ZSCORE:
                        case HYBRID:
                        case WEIGHT_HYBRID:
                        default:
                            value = getNumericalZScoreValue(i, columnNum, obj);
                            break;
                    }
                } else if(columnType == ColumnType.H) {
                    // hybrid column
                    switch(this.normType) {
                        case WOE:
                            value = getHybridWoeValue(i, columnNum, obj, false);
                            break;
                        case WEIGHT_WOE:
                            value = getHybridWoeValue(i, columnNum, obj, true);
                            break;
                        case WOE_ZSCORE:
                        case WOE_ZSCALE:
                            value = getHybridWoeZScoreValue(i, columnNum, obj, false);
                            break;
                        case WEIGHT_WOE_ZSCORE:
                        case WEIGHT_WOE_ZSCALE:
                            value = getHybridWoeZScoreValue(i, columnNum, obj, true);
                            break;
                        case OLD_ZSCALE:
                        case OLD_ZSCORE:
                        case ZSCALE:
                        case ZSCORE:
                        case HYBRID:
                        case WEIGHT_HYBRID:
                        default:
                            throw new IllegalStateException(
                                    "Column type of " + columnName + " is hybrid, but normType is not woe related.");
                    }
                }

                Integer index = entry.getValue();
                if(index != null && index < data.length) {
                    data[currSize + index] = value;
                }
            }
        }
        return data;
    }

    private double getHybridWoeValue(int mtlIndex, Integer columnNum, Object obj, boolean isWeighted) {
        // for hybrid categories, category bin merge is not supported, so we can use
        List<String> binCategories = this.cateColumnNameMapList.get(mtlIndex).get(columnNum);
        Map<String, Integer> cateIndexes = this.cateIndexMapList.get(mtlIndex).get(columnNum);
        Integer binIndex = -1;
        if(obj == null || cateIndexes == null) {
            binIndex = -1;
        } else {
            binIndex = cateIndexes.get(obj.toString());
            if(binIndex == null || binIndex < 0) {
                binIndex = -1;
            }
        }

        List<Double> binBoundaries = this.numerBinBoundaryMapList.get(mtlIndex).get(columnNum);
        if(binIndex != -1) {
            binIndex = binIndex + binBoundaries.size(); // append the first numerical bins
        } else {
            double douVal = BinUtils.parseNumber(obj == null ? null : obj.toString());
            if(Double.isNaN(douVal)) {
                binIndex = binBoundaries.size() + binCategories.size();
            } else {
                binIndex = BinUtils.getBinIndex(binBoundaries, douVal);
            }
        }

        List<Double> binWoes = isWeighted ? this.numerWgtWoeMapList.get(mtlIndex).get(columnNum)
                : this.numerWoeMapList.get(mtlIndex).get(columnNum);
        double value = 0d;
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value;
    }

    private double getNumericalWoeValue(int mtlIndex, Integer columnNum, Object obj, boolean isWeighted) {
        int binIndex = -1;
        if(obj != null) {
            binIndex = BinUtils.getNumericalBinIndex(this.numerBinBoundaryMapList.get(mtlIndex).get(columnNum),
                    obj.toString());
        }
        List<Double> binWoes = isWeighted ? this.numerWgtWoeMapList.get(mtlIndex).get(columnNum)
                : this.numerWoeMapList.get(mtlIndex).get(columnNum);

        double value = 0d;
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value;
    }

    private double getNumericalZScoreValue(int mtlIndex, Integer columnNum, Object obj) {
        double mean = this.numerMeanMapList.get(mtlIndex).get(columnNum);
        double stddev = this.numerStddevMapList.get(mtlIndex).get(columnNum);
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
        double cutoff = Normalizer.checkCutOff(this.cutOffMapList.get(mtlIndex).get(columnNum));
        return Normalizer.computeZScore(rawValue, mean, stddev, cutoff)[0];
    }

    private double getCategoricalPosRateZScoreValue(int mtlIndex, Integer columnNum, Object obj, boolean isOld) {
        double value = 0d;
        Map<String, Double> posRateMapping = this.binPosRateMapList.get(mtlIndex).get(columnNum);
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

        if(isOld) {
            return value;
        }

        double mean = this.numerMeanMapList.get(mtlIndex).get(columnNum);
        double stddev = this.numerStddevMapList.get(mtlIndex).get(columnNum);
        double cutoff = Normalizer.checkCutOff(this.cutOffMapList.get(mtlIndex).get(columnNum));
        return Normalizer.computeZScore(value, mean, stddev, cutoff)[0];
    }

    private double getHybridWoeZScoreValue(int mtlIndex, Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getHybridWoeValue(mtlIndex, columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMapList.get(mtlIndex)
                : this.woeMeanMapList.get(mtlIndex);
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMapList.get(mtlIndex)
                : this.woeStddevMapList.get(mtlIndex);
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double cutoff = Normalizer.checkCutOff(this.cutOffMapList.get(mtlIndex).get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, cutoff)[0];
    }

    private double getNumericalWoeZScoreValue(int mtlIndex, Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getNumericalWoeValue(mtlIndex, columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMapList.get(mtlIndex)
                : this.woeMeanMapList.get(mtlIndex);
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMapList.get(mtlIndex)
                : this.woeStddevMapList.get(mtlIndex);
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double cutoff = Normalizer.checkCutOff(this.cutOffMapList.get(mtlIndex).get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, cutoff)[0];
    }

    private double getCategoricalWoeZScoreValue(int mtlIndex, Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getCategoricalWoeValue(mtlIndex, columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMapList.get(mtlIndex)
                : this.woeMeanMapList.get(mtlIndex);
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMapList.get(mtlIndex)
                : this.woeStddevMapList.get(mtlIndex);
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double cutoff = Normalizer.checkCutOff(cutOffMapList.get(mtlIndex).get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, cutoff)[0];
    }

    private double getCategoricalWoeValue(int mtlIndex, Integer columnNum, Object obj, boolean isWeighted) {
        double value = 0d;
        Map<Integer, Map<String, Double>> mappings = isWeighted ? this.cateWgtWoeMapList.get(mtlIndex)
                : this.cateWoeMapList.get(mtlIndex);
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
     * @return the mtm
     */
    public MultiTaskModel getMtm() {
        return mtm;
    }

}