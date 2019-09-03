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

import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.core.dtrain.nn.NNColumnStats;
import ml.shifu.shifu.util.BinUtils;

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
public class IndependentMTLModel {
    /**
     * MTL graph definition network.
     */
    private MultiTaskLearning mtl;

    /**
     * Normalization type
     */
    private ModelNormalizeConf.NormType normType;

    /**
     * Model version
     */
    private static int version = CommonConstants.MTL_FORMAT_VERSION;

    /**
     * ZScore stddev cutoff value per each column
     */
    private Map<Integer, Double>[] cutOffMapArray;
    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private Map<Integer, String>[] numNameMapArray;
    /**
     * Mapping for (columnNum, binBoundaries) for numberical columns
     */
    private Map<Integer, List<Double>>[] numerBinBoundariesArray;
    /**
     * Mapping for (columnNum, woes) for numberical columns; for hybrid, woe bins for both numberical and categorical
     * bins; last one in weightedWoes is for missing value bin
     */
    private Map<Integer, List<Double>>[] numerWoesArray;
    /**
     * Mapping for (columnNum, wgtWoes) for numberical columns; for hybrid, woe bins for both numberical and categorical
     * bins; last one in weightedBinWoes is for missing value bin
     */
    private Map<Integer, List<Double>>[] numerWgtWoesArray;
    /**
     * Mapping for (columnNum, mean) for all columns
     */
    private Map<Integer, Double>[] numerMeanMapArray;
    /**
     * Mapping for (columnNum, stddev) for all columns
     */
    private Map<Integer, Double>[] numerStddevMapArray;
    /**
     * Mapping for (columnNum, woeMean) for all columns
     */
    private Map<Integer, Double>[] woeMeanMapArray;
    /**
     * Mapping for (columnNum, woeStddev) for all columns
     */
    private Map<Integer, Double>[] woeStddevMapArray;
    /**
     * Mapping for (columnNum, weightedWoeMean) for all columns
     */
    private Map<Integer, Double>[] wgtWoeMeanMapArray;
    /**
     * Mapping for (columnNum, weightedWoeStddev) for all columns
     */
    private Map<Integer, Double>[] wgtWoeStddevMapArray;
    /**
     * Mapping for (ColumnNum, index in double[] array)
     */
    private Map<Integer, Integer>[] columnNumIndexMappingArray;

    /**
     * number of tasks.
     */
    private int taskNumber;

    private IndependentMTLModel(MultiTaskLearning mtnn, ModelNormalizeConf.NormType normType,
            Map<Integer, Double>[] cutOffMapArray, Map<Integer, String>[] numNameMapArray,
            Map<Integer, List<Double>>[] numerBinBoundariesArray, Map<Integer, List<Double>>[] numerWoesArray,
            Map<Integer, List<Double>>[] numerWgtWoesArray, Map<Integer, Double>[] numerMeanMapArray,
            Map<Integer, Double>[] numerStddevMapArray, Map<Integer, Double>[] woeMeanMapArray,
            Map<Integer, Double>[] woeStddevMapArray, Map<Integer, Double>[] wgtWoeMeanMapArray,
            Map<Integer, Double>[] wgtWoeStddevMapArray, Map<Integer, Integer>[] columnNumIndexMappingArray,
            int taskNumber) {
        this.mtl = mtnn;
        this.normType = normType;
        this.cutOffMapArray = cutOffMapArray;
        this.numNameMapArray = numNameMapArray;
        this.numerBinBoundariesArray = numerBinBoundariesArray;
        this.numerWoesArray = numerWoesArray;
        this.numerWgtWoesArray = numerWgtWoesArray;
        this.numerMeanMapArray = numerMeanMapArray;
        this.numerStddevMapArray = numerStddevMapArray;
        this.woeMeanMapArray = woeMeanMapArray;
        this.woeStddevMapArray = woeStddevMapArray;
        this.wgtWoeMeanMapArray = wgtWoeMeanMapArray;
        this.wgtWoeStddevMapArray = wgtWoeStddevMapArray;
        this.columnNumIndexMappingArray = columnNumIndexMappingArray;
        this.taskNumber = taskNumber;
    }

    /**
     * Compute forward score according to data inputs
     *
     * @param denseInputs,
     *            the dense inputs
     */
    public double[] realCompute(double[] denseInputs) {
        return (this.mtl.forward(denseInputs));
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
     * @param dataMap
     *            {@code dataMap} for (columnName, value), numberic value can be double/String, categorical feature can
     *            be int(index) or category value. if not set or set to null, such feature will be treated as missing
     *            value. For numberical value, if it cannot be parsed successfully, it will also be treated as missing.
     * @return score output for wide and deep model
     */
    public double[] compute(Map<String, Object> dataMap) {
        return realCompute(getDenseInputs(dataMap));
    }

    public double[] compute(double[] data) {
        if(data == null) {
            return null;
        }
        return realCompute(getDenseInputs(data));
    }

    private double[] getDenseInputs(Map<String, Object> dataMap) {
        // Get dense inputs
        List<Double> tmpList = new ArrayList<>();
        Object value;

        for(int i = 0; i < taskNumber; i++) {
            Map<Integer, Integer> columnNumIndexMapping = this.columnNumIndexMappingArray[i];
            Map<Integer, String> numNameMap = this.numNameMapArray[i];

            for(Map.Entry<Integer, Integer> entry: columnNumIndexMapping.entrySet()) {
                int columnNum = entry.getKey();
                value = dataMap.get(numNameMap.get(columnNum));
                if(value != null) {
                    tmpList.add(normalize(i, columnNum, value, this.normType));
                } else {
                    tmpList.add(getMissingNumericalValue(i, columnNum));
                }
            }
        }

        return getDoubles(tmpList);
    }

    // covert list to array.
    private double[] getDoubles(List<Double> tmpList) {
        double[] result = new double[tmpList.size()];
        for(int i = 0; i < tmpList.size(); i++) {
            result[i] = tmpList.get(i);
        }
        return result;
    }

    private double[] getDenseInputs(double[] data) {
        List<Double> tmpList = new ArrayList<>();
        for(int i = 0; i < taskNumber; i++) {
            for(Map.Entry<Integer, Integer> entry: this.columnNumIndexMappingArray[i].entrySet()) {
                tmpList.add(data[entry.getValue()]);
            }
        }
        return getDoubles(tmpList);
    }

    private double normalize(int ccsNum, int columnNum, Object obj, ModelNormalizeConf.NormType normType) {
        double value;
        // numberical column
        switch(this.normType) {
            case WOE:
                value = getNumericalWoeValue(ccsNum, columnNum, obj, false);
                break;
            case WEIGHT_WOE:
                value = getNumericalWoeValue(ccsNum, columnNum, obj, true);
                break;
            case WOE_ZSCORE:
            case WOE_ZSCALE:
                value = getNumericalWoeZScoreValue(ccsNum, columnNum, obj, false);
                break;
            case WEIGHT_WOE_ZSCORE:
            case WEIGHT_WOE_ZSCALE:
                value = getNumericalWoeZScoreValue(ccsNum, columnNum, obj, true);
                break;
            case OLD_ZSCALE:
            case OLD_ZSCORE:
            case ZSCALE:
            case ZSCORE:
            case HYBRID:
            case WEIGHT_HYBRID:
            default:
                value = getNumericalZScoreValue(ccsNum, columnNum, obj);
                break;
        }
        return value;
    }

    private double getMissingNumericalValue(int ccsNum, int columnId) {
        return defaultMissingValue(this.numerMeanMapArray[ccsNum].get(columnId));
    }

    private double defaultMissingValue(Double mean) {
        Double defaultValue = mean == null ? 0 : mean;
        return defaultValue.doubleValue();
    }

    private double getNumericalWoeValue(int ccsNum, Integer columnNum, Object obj, boolean isWeighted) {
        int binIndex = -1;
        if(obj != null) {
            binIndex = BinUtils.getNumericalBinIndex(this.numerBinBoundariesArray[ccsNum].get(columnNum),
                    obj.toString());
        }
        List<Double> binWoes = isWeighted ? this.numerWgtWoesArray[ccsNum].get(columnNum)
                : this.numerWoesArray[ccsNum].get(columnNum);

        Double value;
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value.doubleValue();
    }

    private double getNumericalWoeZScoreValue(int ccsNum, Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getNumericalWoeValue(ccsNum, columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.wgtWoeMeanMapArray[ccsNum] : this.woeMeanMapArray[ccsNum];
        Map<Integer, Double> woeStddevs = isWeighted ? this.wgtWoeStddevMapArray[ccsNum]
                : this.woeStddevMapArray[ccsNum];
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double realCutoff = Normalizer.checkCutOff(this.cutOffMapArray[ccsNum].get(columnNum));
        return Normalizer.computeZScore(woe, mean, stddev, realCutoff)[0].doubleValue();
    }

    private double getNumericalZScoreValue(int ccsNum, Integer columnNum, Object obj) {
        double mean = this.numerMeanMapArray[ccsNum].get(columnNum);
        double stddev = this.numerStddevMapArray[ccsNum].get(columnNum);
        double rawValue;
        if(obj == null || obj.toString().length() == 0) {
            rawValue = defaultMissingValue(mean);
        } else {
            try {
                rawValue = Double.parseDouble(obj.toString());
            } catch (Exception e) {
                rawValue = defaultMissingValue(mean);
            }
        }
        double realCutoff = Normalizer.checkCutOff(this.cutOffMapArray[ccsNum].get(columnNum));
        return Normalizer.computeZScore(rawValue, mean, stddev, realCutoff)[0].doubleValue();
    }

    /**
     * Load model instance from input stream which is saved in MTLOutput for specified binary format.
     *
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @return the mtl model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */
    public static IndependentMTLModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, true);
    }

    /**
     * Load model instance from input stream which is saved in MTLOutput for specified binary format.
     *
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @param isRemoveNameSpace,
     *            is remove name space or not
     * @return the mtl model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */

    public static IndependentMTLModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        DataInputStream dis;
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
        IndependentMTLModel.setVersion(version);
        // Reserved two double field, one double field and one string field
        dis.readDouble();
        dis.readDouble();
        dis.readDouble();
        dis.readUTF();

        // read normStr
        String normStr = StringUtils.readString(dis);
        ModelNormalizeConf.NormType normType = ModelNormalizeConf.NormType
                .valueOf(normStr != null ? normStr.toUpperCase() : null);

        int taskNumber = dis.readInt();

        // for all features
        Map<Integer, String>[] numNameMapArray = new Map[taskNumber];
        // for numerical features
        Map<Integer, List<Double>>[] numerBinBoundariesArray = new Map[taskNumber];
        Map<Integer, List<Double>>[] numerWoesArray = new Map[taskNumber];
        Map<Integer, List<Double>>[] numerWgtWoesArray = new Map[taskNumber];
        // for all features
        Map<Integer, Double>[] numerMeanMapArray = new Map[taskNumber];
        Map<Integer, Double>[] numerStddevMapArray = new Map[taskNumber];
        Map<Integer, Double>[] woeMeanMapArray = new Map[taskNumber];
        Map<Integer, Double>[] woeStddevMapArray = new Map[taskNumber];
        Map<Integer, Double>[] wgtWoeMeanMapArray = new Map[taskNumber];
        Map<Integer, Double>[] wgtWoeStddevMapArray = new Map[taskNumber];
        Map<Integer, Double>[] cutoffMapArray = new Map[taskNumber];
        Map<Integer, Integer>[] columnMappingArray = new Map[taskNumber];

        for(int t = 0; t < taskNumber; t++) {
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

            for(int i = 0; i < columnSize; i++) {
                NNColumnStats cs = new NNColumnStats();
                cs.readFields(dis);

                List<Double> binWoes = cs.getBinCountWoes();
                List<Double> binWgtWoes = cs.getBinWeightWoes();
                int columnNum = cs.getColumnNum();

                if(isRemoveNameSpace) {
                    // remove name-space in column name to make it be called by simple name
                    numNameMap.put(columnNum, StringUtils.getSimpleColumnName(cs.getColumnName()));
                } else {
                    numNameMap.put(columnNum, cs.getColumnName());
                }

                if(cs.isNumerical() || cs.isHybrid()) {
                    numerBinBoundaries.put(columnNum, cs.getBinBoundaries());
                    numerWoes.put(columnNum, binWoes);
                    numerWgtWoes.put(columnNum, binWgtWoes);
                }

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
            for(int i = 0; i < columnMappingSize; i++) {
                columnMapping.put(dis.readInt(), dis.readInt());
            }

            // add different maps to different mapArrays.
            numNameMapArray[t] = numNameMap;
            numerBinBoundariesArray[t] = numerBinBoundaries;
            numerWoesArray[t] = numerWoes;
            numerWgtWoesArray[t] = numerWgtWoes;
            numerMeanMapArray[t] = numerMeanMap;
            numerStddevMapArray[t] = numerStddevMap;
            woeMeanMapArray[t] = woeMeanMap;
            woeStddevMapArray[t] = woeStddevMap;
            wgtWoeMeanMapArray[t] = wgtWoeMeanMap;
            wgtWoeStddevMapArray[t] = wgtWoeStddevMap;
            cutoffMapArray[t] = cutoffMap;
            columnMappingArray[t] = columnMapping;
        }

        MultiTaskLearning mtnn = new MultiTaskLearning();
        mtnn.readFields(dis);
        return new IndependentMTLModel(mtnn, normType, cutoffMapArray, numNameMapArray, numerBinBoundariesArray,
                numerWoesArray, numerWgtWoesArray, numerMeanMapArray, numerStddevMapArray, woeMeanMapArray,
                woeStddevMapArray, wgtWoeMeanMapArray, wgtWoeStddevMapArray, columnMappingArray, taskNumber);
    }

    public static int getVersion() {
        return version;
    }

    public static void setVersion(int version) {
        IndependentMTLModel.version = version;
    }

    public MultiTaskLearning getMtl() {
        return mtl;
    }

    public void setMtl(MultiTaskLearning mtl) {
        this.mtl = mtl;
    }
}
