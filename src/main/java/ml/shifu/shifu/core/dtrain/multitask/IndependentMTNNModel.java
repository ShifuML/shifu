package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.core.dtrain.nn.NNColumnStats;
import ml.shifu.shifu.util.BinUtils;
import ml.shifu.shifu.util.Constants;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * @author haillu
 */
public class IndependentMTNNModel {
    /**
     * MTL graph definition network.
     */
    private MultiTaskNN mtnn;

    /**
     * Normalization type
     */
    private ModelNormalizeConf.NormType normType;

    /**
     * Model version
     */
    private static int version = CommonConstants.MTL_FORMAT_VERSION;

    /**
     * Mapping for (ColumnNum, Map(Category, CategoryIndex) for categorical feature
     */
    private Map<Integer, Map<String, Integer>> cateIndexMap;
    /**
     * ZScore stddev cutoff value per each column
     */
    private Map<Integer, Double> cutOffMap;
    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private Map<Integer, String> numNameMap;
    /**
     * Mapping for (columnNum, binBoundaries) for numberical columns
     */
    private Map<Integer, List<Double>> numerBinBoundaries;
    /**
     * Mapping for (columnNum, woes) for numberical columns; for hybrid, woe bins for both numberical and categorical
     * bins; last one in weightedWoes is for missing value bin
     */
    private Map<Integer, List<Double>> numerWoes;
    /**
     * Mapping for (columnNum, wgtWoes) for numberical columns; for hybrid, woe bins for both numberical and categorical
     * bins; last one in weightedBinWoes is for missing value bin
     */
    private Map<Integer, List<Double>> numerWgtWoes;
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
     * Mapping for (ColumnNum, index in double[] array)
     */
    private Map<Integer, Integer> columnNumIndexMapping;

    private IndependentMTNNModel(MultiTaskNN mtnn, ModelNormalizeConf.NormType normType, Map<Integer, Map<String, Integer>> cateIndexMap,
                                 Map<Integer, Double> cutOffMap, Map<Integer, String> numNameMap, Map<Integer, List<Double>> numerBinBoundaries,
                                 Map<Integer, List<Double>> numerWoes, Map<Integer, List<Double>> numerWgtWoes, Map<Integer, Double> numerMeanMap,
                                 Map<Integer, Double> numerStddevMap, Map<Integer, Double> woeMeanMap, Map<Integer, Double> woeStddevMap,
                                 Map<Integer, Double> wgtWoeMeanMap, Map<Integer, Double> wgtWoeStddevMap, Map<Integer, Integer> columnNumIndexMapping) {
        this.mtnn = mtnn;
        this.normType = normType;
        this.cateIndexMap = cateIndexMap;
        this.cutOffMap = cutOffMap;
        this.numNameMap = numNameMap;
        this.numerBinBoundaries = numerBinBoundaries;
        this.numerWoes = numerWoes;
        this.numerWgtWoes = numerWgtWoes;
        this.numerMeanMap = numerMeanMap;
        this.numerStddevMap = numerStddevMap;
        this.woeMeanMap = woeMeanMap;
        this.woeStddevMap = woeStddevMap;
        this.wgtWoeMeanMap = wgtWoeMeanMap;
        this.wgtWoeStddevMap = wgtWoeStddevMap;
        this.columnNumIndexMapping = columnNumIndexMapping;
    }

    /**
     * Compute forward score according to data inputs
     *
     * @param denseInputs, the dense inputs
     */
    public double[] realCompute(double[] denseInputs) {
        return (this.mtnn.forward(denseInputs));
    }


    /**
     * Given {@code dataMap} with format (columnName, value), compute score values of wide and deep model.
     *
     * <p>
     * No any alert or exception if your {@code dataMap} doesn't contain features included in the model, such case will
     * be treated as missing value case. Please make sure feature names in keys of {@code dataMap} are consistent with
     * names in model.
     *
     * <p>
     * In {@code dataMap}, numberical value can be (String, Double) format or (String, String) format, they will all be
     * parsed to Double; categorical value are all converted to (String, String). If value not in our categorical list,
     * it will also be treated as missing value.
     *
     * <p>
     * In {@code dataMap}, data should be raw value and normalization is computed inside according to {@link #normType}
     * and stats information in such model.
     *
     * @param dataMap {@code dataMap} for (columnName, value), numberic value can be double/String, categorical feature can
     *                be int(index) or category value. if not set or set to null, such feature will be treated as missing
     *                value. For numberical value, if it cannot be parsed successfully, it will also be treated as missing.
     * @return score output for wide and deep model
     */
    public double[] compute(Map<String, Object> dataMap) {
        return realCompute(getDenseInputs(dataMap));
    }

    public double[] compute(double[] data) {
        if (data == null) {
            return null;
        }
        return realCompute(getDenseInputs(data));
    }

    private double[] getDenseInputs(Map<String, Object> dataMap) {
        // Get dense inputs
        double[] result = new double[this.columnNumIndexMapping.size()];
        Object value;
        int index = 0;
        for (Map.Entry<Integer, Integer> entry : this.columnNumIndexMapping.entrySet()) {
            int columnNum = entry.getKey();
            value = dataMap.get(numNameMap.get(columnNum));
            if (value != null) {
                result[index] = normalize(columnNum, value, this.normType);
            } else {
                result[index] = getMissingNumericalValue(columnNum);
            }
            index++;
        }
        return result;
    }

    private double[] getDenseInputs(double[] data) {
        double[] result = new double[this.columnNumIndexMapping.size()];
        int index = 0;
        for (Map.Entry<Integer, Integer> entry : this.columnNumIndexMapping.entrySet()) {
            result[index] = data[entry.getValue()];
            index++;
        }
        return result;
    }

    private double normalize(int columnNum, Object obj, ModelNormalizeConf.NormType normType) {
        double value;
        // numberical column
        switch (this.normType) {
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
        return value;
    }


    private Object getValueByColumnId(int columnId, Map<String, Object> dataMap) {
        return dataMap.get(this.numNameMap.get(columnId));
    }

    private double getMissingNumericalValue(int columnId) {
        return defaultMissingValue(this.numerMeanMap.get(columnId));
    }

    private double defaultMissingValue(Double mean) {
        Double defaultValue = mean == null ? 0 : mean;
        return defaultValue.doubleValue();
    }

    private double getNumericalWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        int binIndex = -1;
        if (obj != null) {
            binIndex = BinUtils.getNumericalBinIndex(this.numerBinBoundaries.get(columnNum), obj.toString());
        }
        List<Double> binWoes = isWeighted ? this.numerWgtWoes.get(columnNum) : this.numerWoes.get(columnNum);

        Double value;
        if (binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value.doubleValue();
    }

    private double getNumericalWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getNumericalWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMap : this.woeMeanMap;
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMap : this.woeStddevMap;
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double realCutoff = Normalizer.checkCutOff(this.cutOffMap.get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, realCutoff)[0].doubleValue();
    }

    private double getNumericalZScoreValue(Integer columnNum, Object obj) {
        double mean = this.numerMeanMap.get(columnNum);
        double stddev = this.numerStddevMap.get(columnNum);
        double rawValue;
        if (obj == null || obj.toString().length() == 0) {
            rawValue = defaultMissingValue(mean);
        } else {
            try {
                rawValue = Double.parseDouble(obj.toString());
            } catch (Exception e) {
                rawValue = defaultMissingValue(mean);
            }
        }
        double realCutoff = Normalizer.checkCutOff(this.cutOffMap.get(columnNum));
        return Normalizer.computeZScore(rawValue, mean, stddev, realCutoff)[0].doubleValue();
    }

    private Object[] map2array(Map<Integer, Integer> map) {
        return new ArrayList<>(map.values()).toArray();
    }

    /**
     * Load model instance from input stream which is saved in MTNNOutput for specified binary format.
     *
     * @param input the input stream, flat input stream or gzip input stream both OK
     * @return the mtl model instance
     * @throws IOException any IOException in de-serialization.
     */
    public static IndependentMTNNModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, true);
    }

    /**
     * Load model instance from input stream which is saved in MTNNOutput for specified binary format.
     *
     * @param input              the input stream, flat input stream or gzip input stream both OK
     * @param isRemoveNameSpace, is remove name space or not
     * @return the mtl model instance
     * @throws IOException any IOException in de-serialization.
     */

    public static IndependentMTNNModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        DataInputStream dis;
        // check if gzip or not
        try {
            byte[] header = new byte[2];
            BufferedInputStream bis = new BufferedInputStream(input);
            bis.mark(2);
            int result = bis.read(header);
            bis.reset();
            int ss = (header[0] & 0xff) | ((header[1] & 0xff) << 8);
            if (result != -1 && ss == GZIPInputStream.GZIP_MAGIC) {
                dis = new DataInputStream(new GZIPInputStream(bis));
            } else {
                dis = new DataInputStream(bis);
            }
        } catch (java.io.IOException e) {
            dis = new DataInputStream(input);
        }

        int version = dis.readInt();
        IndependentMTNNModel.setVersion(version);
        // Reserved two double field, one double field and one string field
        dis.readDouble();
        dis.readDouble();
        dis.readDouble();
        dis.readUTF();

        // read normStr
        String normStr = StringUtils.readString(dis);
        ModelNormalizeConf.NormType normType = ModelNormalizeConf.NormType.valueOf(normStr != null ? normStr.toUpperCase() : null);

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
        Map<Integer, Map<String, Integer>> cateIndexMapping = new HashMap<>(columnSize);
        for (int i = 0; i < columnSize; i++) {
            NNColumnStats cs = new NNColumnStats();
            cs.readFields(dis);

            List<Double> binWoes = cs.getBinCountWoes();
            List<Double> binWgtWoes = cs.getBinWeightWoes();
            int columnNum = cs.getColumnNum();

            if (isRemoveNameSpace) {
                // remove name-space in column name to make it be called by simple name
                numNameMap.put(columnNum, StringUtils.getSimpleColumnName(cs.getColumnName()));
            } else {
                numNameMap.put(columnNum, cs.getColumnName());
            }

            // for categorical features
            Map<String, Integer> cateIndexMap = new HashMap<>(cs.getBinCategories().size());

            if (cs.isCategorical() || cs.isHybrid()) {
                List<String> binCategories = cs.getBinCategories();

                for (int j = 0; j < binCategories.size(); j++) {
                    String currCate = binCategories.get(j);
                    if (currCate.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                        // merged category should be flatten, use own split function to avoid depending on guava jar in
                        // prediction
                        String[] splits = StringUtils.split(currCate, Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
                        for (String str : splits) {
                            cateIndexMap.put(str, j);
                        }
                    } else {
                        cateIndexMap.put(currCate, j);
                    }
                }
            }

            if (cs.isNumerical() || cs.isHybrid()) {
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
        }

        int columnMappingSize = dis.readInt();
        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnMappingSize, 1f);
        for (int i = 0; i < columnMappingSize; i++) {
            columnMapping.put(dis.readInt(), dis.readInt());
        }

        MultiTaskNN mtnn = new MultiTaskNN();
        mtnn.readFields(dis);
        return new IndependentMTNNModel(mtnn, normType, cateIndexMapping, cutoffMap, numNameMap,
                numerBinBoundaries, numerWoes, numerWgtWoes, numerMeanMap, numerStddevMap, woeMeanMap, woeStddevMap,
                wgtWoeMeanMap, wgtWoeStddevMap, columnMapping);
    }


    public static int getVersion() {
        return version;
    }

    public static void setVersion(int version) {
        IndependentMTNNModel.version = version;
    }

    public MultiTaskNN getMtnn() {
        return mtnn;
    }

    public void setMtnn(MultiTaskNN mtnn) {
        this.mtnn = mtnn;
    }
}
