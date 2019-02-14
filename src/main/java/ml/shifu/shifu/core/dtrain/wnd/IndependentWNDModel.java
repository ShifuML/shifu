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
package ml.shifu.shifu.core.dtrain.wnd;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.Normalizer;
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

import static ml.shifu.shifu.core.dtrain.CommonConstants.WND_FORMAT_VERSION;

/**
 * {@link IndependentWNDModel} is a light WND engine to predict WND model, the only dependency is shifu, guagua and
 * encog-core jars.
 *
 * <p>
 * {@link #compute(Map)} are the API called for prediction.
 *
 * <p>
 * @author : Wu Devin (haifwu@paypal.com)
 */
public class IndependentWNDModel {

    /**
     * WideAndDeep graph definition network.
     */
    private WideAndDeep wnd;

    /**
     * Normalization type
     */
    private NormType normType;

    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private Map<Integer, String> columnIdNameMap;

    /**
     * Mapping for (ColumnNum, Map(Category, CategoryIndex) for categorical feature
     */
    private Map<Integer, Map<String, Integer>> cateIndexMap;

    /**
     * Mapping for (columnNum, binBoundaries) for numberical columns
     */
    private Map<Integer, List<Float>> numberBinBoundaries;

    /**
     * Mapping for (columnNum, woes) for numberical columns; for hybrid, woe bins for both numberical and categorical
     * bins; last one in weightedWoes is for missing value bin
     */
    private Map<Integer, List<Float>> numberWoes;

    /**
     * Mapping for (columnNum, wgtWoes) for numberical columns; for hybrid, woe bins for both numberical and categorical
     * bins; last one in weightedBinWoes is for missing value bin
     */
    private Map<Integer, List<Float>> numberWgtWoes;

    /**
     * ZScore stddev cutoff
     */
    private double cutOff;

    /**
     * Mapping for (columnNum, mean) for all columns
     */
    private Map<Integer, Float> numberMeanMap;

    /**
     * Mapping for (columnNum, stddev) for all columns
     */
    private Map<Integer, Float> numberStddevMap;

    /**
     * Mapping for (columnNum, woeMean) for all columns
     */
    private Map<Integer, Float> woeMeanMap;

    /**
     * Mapping for (columnNum, woeStddev) for all columns
     */
    private Map<Integer, Float> woeStddevMap;

    /**
     * Mapping for (columnNum, weightedWoeMean) for all columns
     */
    private Map<Integer, Float> wgtWoeMeanMap;

    /**
     * Mapping for (columnNum, weightedWoeStddev) for all columns
     */
    private Map<Integer, Float> wgtWoeStddevMap;
    
    
    private IndependentWNDModel(WideAndDeep wideAndDeep, NormType normType, double cutOff,
                                Map<Integer, String> columnIdNameMap,
                                Map<Integer, Map<String, Integer>> cateIndexMap, 
                                Map<Integer, List<Float>> numberBinBoundaries,
                                Map<Integer, List<Float>> numberWoes,  Map<Integer, List<Float>> numberWgtWoes,
                                Map<Integer, Float> numberMeanMap,
                                Map<Integer, Float> numberStddevMap, Map<Integer, Float> woeMeanMap,
                                Map<Integer, Float> woeStddevMap, Map<Integer, Float> wgtWoeMeanMap,
                                Map<Integer, Float> wgtWoeStddevMap) {
        this.wnd = wideAndDeep;
        this.normType = normType;
        this.columnIdNameMap = columnIdNameMap;
        this.cateIndexMap = cateIndexMap;
        this.numberBinBoundaries = numberBinBoundaries;
        this.numberWoes = numberWoes;
        this.numberWgtWoes = numberWgtWoes;
        this.cutOff = cutOff;
        this.numberMeanMap = numberMeanMap;
        this.numberStddevMap = numberStddevMap;
        this.woeMeanMap = woeMeanMap;
        this.woeStddevMap = woeStddevMap;
        this.wgtWoeMeanMap = wgtWoeMeanMap;
        this.wgtWoeStddevMap = wgtWoeStddevMap;
    }


    /**
     * Compute logits according to data inputs
     * @param denseInputs, the dense inputs for deep model, numerical values
     * @param embedInputs, the embed inputs for deep model, category values
     * @param wideInputs, the wide model inputs, category values
     * @return model score of the inputs.
     */
    public float[] compute(float[] denseInputs, List<SparseInput> embedInputs, List<SparseInput> wideInputs){
        return this.wnd.forward(denseInputs, embedInputs, wideInputs);
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
     * In {@code dataMap}, numberical value can be (String, Float) format or (String, String) format, they will all be
     * parsed to Float; categorical value are all converted to (String, String). If value not in our categorical list,
     * it will also be treated as missing value.
     *
     * <p>
     * In {@code dataMap}, data should be raw value and normalization is computed inside according to {@link #normType}
     * and stats information in such model.
     *
     * @param dataMap
     *            {@code dataMap} for (columnName, value), numberic value can be float/String, categorical feature can
     *            be int(index) or category value. if not set or set to null, such feature will be treated as missing
     *            value. For numberical value, if it cannot be parsed successfully, it will also be treated as missing.
     * @return score output for wide and deep model
     */
    public float[] compute(Map<String, Object> dataMap) {
        return compute(getDenseInputs(dataMap), getEmbedInputs(dataMap), getWideInputs(dataMap));
    }

    private float[] getDenseInputs(Map<String, Object> dataMap) {
        // Get dense inputs
        float[] numbericalValues = new float[this.wnd.getDenseColumnIds().size()];
        Object value;
        for (int i = 0; i < this.wnd.getDenseColumnIds().size(); i++){
            value = getValueByColumnId(this.wnd.getDenseColumnIds().get(i), dataMap);
            if (value != null) {
                numbericalValues[i] = normalize(this.wnd.getDenseColumnIds().get(i), value, this.normType);
            } else {
                numbericalValues[i] = getMissingnumbericalValue(this.wnd.getDenseColumnIds().get(i));
            }
        }
        return numbericalValues;
    }

    private float normalize(int columnNum, Object obj, NormType normType) {
        float value;
        // numberical column
        switch(this.normType) {
            case WOE:
                value = getnumbericalWoeValue(columnNum, obj, false);
                break;
            case WEIGHT_WOE:
                value = getnumbericalWoeValue(columnNum, obj, true);
                break;
            case WOE_ZSCORE:
            case WOE_ZSCALE:
                value = getnumbericalWoeZScoreValue(columnNum, obj, false);
                break;
            case WEIGHT_WOE_ZSCORE:
            case WEIGHT_WOE_ZSCALE:
                value = getnumbericalWoeZScoreValue(columnNum, obj, true);
                break;
            case OLD_ZSCALE:
            case OLD_ZSCORE:
            case ZSCALE:
            case ZSCORE:
            case HYBRID:
            case WEIGHT_HYBRID:
            default:
                value = getnumbericalZScoreValue(columnNum, obj);
                break;
        }
        return value;
    }
    
    private Object getValueByColumnId(int columnId, Map<String, Object> dataMap) {
        return dataMap.get(this.columnIdNameMap.get(columnId));
    }

    private List<SparseInput> getEmbedInputs(Map<String, Object> dataMap) {
        List<SparseInput> embedInputs = new ArrayList<>();
        Object value;
        for(Integer columnId: this.wnd.getEmbedColumnIds()) {
            value = getValueByColumnId(columnId, dataMap);
            if (value != null) {
                embedInputs.add(new SparseInput(columnId, getValueIndex(columnId, value.toString())));
            } else {
                // when the value missing
                embedInputs.add(new SparseInput(columnId, getMissingTypeCategory(columnId)));
            }
        }
        return embedInputs;
    }
    
    private int getMissingTypeCategory(int columnId) {
        //TODO if this right? Current return +1 of the last index
        return this.cateIndexMap.get(columnId).values().size();
    }
    
    private float getMissingnumbericalValue(int columnId) {
        // TODO if this right? Currently return the mean value
        return defaultMissingValue(this.numberMeanMap.get(columnId));
    }

    private float defaultMissingValue(Float mean) {
        return mean == null ? 0 : mean;
    }
    
    private int getValueIndex(int columnId, String value) {
        return this.cateIndexMap.get(columnId).get(value);
    }

    private List<SparseInput> getWideInputs(Map<String, Object> dataMap) {
        List<SparseInput> wideInputs = new ArrayList<>();
        Object value;
        for(Integer columnId: this.wnd.getWideColumnIds()) {
            value = getValueByColumnId(columnId, dataMap);
            if (value != null) {
                wideInputs.add(new SparseInput(columnId, getValueIndex(columnId, value.toString())));
            } else {
                // when the value missing
                wideInputs.add(new SparseInput(columnId, getMissingTypeCategory(columnId)));
            }
        }
        return wideInputs;
    }

    /**
     * Load model instance from input stream which is saved in WNDOutput for specified binary format.
     *
     * @param input
     *            the input stream, flat input stream or gzip input stream both OK
     * @return the WideAndDeep model instance
     * @throws IOException
     *             any IOException in de-serialization.
     */
    public static IndependentWNDModel loadFromStream(InputStream input) throws IOException {
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

        assert WND_FORMAT_VERSION == dis.readInt();
        //TODO assert this
        String algorithm = dis.readUTF();

        WideAndDeep wideAndDeep = PersistWideAndDeep.load(dis);
        NormType normType = NormType.valueOf(dis.readUTF());
        double cutOff = dis.readDouble();
        return buildIndependentWNDModel(wideAndDeep, normType, cutOff);
    }

    private static IndependentWNDModel buildIndependentWNDModel(WideAndDeep wideAndDeep, NormType normType,
                                                                double cutOff) {
        Map<Integer, String> columnIdNameMap = new HashMap<>();
        Map<Integer, Map<String, Integer>> cateIndexMap = new HashMap<>();
        Map<Integer, List<Float>> numberBinBoundaries = new HashMap<>();
        Map<Integer, List<Float>> numberWoes = new HashMap<>();
        Map<Integer, List<Float>> numberWgtWoes = new HashMap<>();
        Map<Integer, Float> numberMeanMap = new HashMap<>();
        Map<Integer, Float> numberStddevMap = new HashMap<>();
        Map<Integer, Float> woeMeanMap = new HashMap<>();
        Map<Integer, Float> woeStddevMap = new HashMap<>();
        Map<Integer, Float> wgtWoeMeanMap = new HashMap<>();
        Map<Integer, Float> wgtWoeStddevMap = new HashMap<>();

        List<ColumnConfig> columnConfigList = wideAndDeep.getColumnConfigList();
        for(ColumnConfig columnConfig: columnConfigList) {
            // build column Id -> name map
            columnIdNameMap.put(columnConfig.getColumnNum(), columnConfig.getColumnName());

            // for category value: build column id -> { category -> index } map
            Map<String, Integer> indexMap = new HashMap<>(1);
            if(columnConfig.getBinCategory() != null) {
                for(int i = 0; i < columnConfig.getBinCategory().size(); i++) {
                    indexMap.put(columnConfig.getBinCategory().get(i), i);
                }
            }
            cateIndexMap.put(columnConfig.getColumnNum(), indexMap);

            // for numerical value: build column id -> bin boundaries
            List<Float> binBoundaries = new ArrayList<>();
            if(columnConfig.getBinBoundary() != null) {
                for(Double boundary: columnConfig.getBinBoundary()) {
                    binBoundaries.add(boundary.floatValue());
                }
                numberBinBoundaries.put(columnConfig.getColumnNum(), binBoundaries);
            }

            // for numerical value: build number -> Woes list map
            List<Float> woes = new ArrayList<>();
            if(columnConfig.getBinCountWoe() != null) {
                for(Double woe: columnConfig.getBinCountWoe()) {
                    woes.add(woe.floatValue());
                }
            }
            numberWoes.put(columnConfig.getColumnNum(), woes);

            // for numerical value: build number ->  wgt woes list map
            List<Float> wgtWoes = new ArrayList<>();
            if(columnConfig.getBinWeightedWoe() != null) {
                for(Double woe: columnConfig.getBinWeightedWoe()) {
                    wgtWoes.add(woe.floatValue());
                }
            }
            numberWgtWoes.put(columnConfig.getColumnNum(), wgtWoes);

            // for numerical value: build number -> mean Map
            if(columnConfig.getMean() != null) {
                numberMeanMap.put(columnConfig.getColumnNum(), columnConfig.getMean().floatValue());
            }

            // for numerical value: build number -> stddev map
            if(columnConfig.getStdDev() != null) {
                numberMeanMap.put(columnConfig.getColumnNum(), columnConfig.getStdDev().floatValue());
            }

            // for numerical value: build number -> woe mean stddev
            double[] woeMeanAndStdDev= Normalizer.calculateWoeMeanAndStdDev(columnConfig, false);
            woeMeanMap.put(columnConfig.getColumnNum(), Double.valueOf(woeMeanAndStdDev[0]).floatValue());
            woeStddevMap.put(columnConfig.getColumnNum(), Double.valueOf(woeMeanAndStdDev[1]).floatValue());

            // for numerical value: build number -> wgt woe mean stddev
            double[] wgtWoeMeanAndStdDev= Normalizer.calculateWoeMeanAndStdDev(columnConfig, true);
            wgtWoeMeanMap.put(columnConfig.getColumnNum(), Double.valueOf(wgtWoeMeanAndStdDev[0]).floatValue());
            wgtWoeMeanMap.put(columnConfig.getColumnNum(), Double.valueOf(wgtWoeMeanAndStdDev[1]).floatValue());
        }
        return new IndependentWNDModel(wideAndDeep, normType, cutOff, columnIdNameMap, cateIndexMap,
                numberBinBoundaries, numberWoes, numberWgtWoes, numberMeanMap, numberStddevMap, woeMeanMap,
                woeStddevMap, wgtWoeMeanMap, wgtWoeStddevMap);
    }

    private float getnumbericalWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        int binIndex = -1;
        if(obj != null) {
            binIndex = BinUtils.getNumericalBinIndexFloat(this.numberBinBoundaries.get(columnNum), obj.toString());
        }
        List<Float> binWoes = isWeighted ? this.numberWgtWoes.get(columnNum) : this.numberWoes.get(columnNum);

        float value;
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value;
    }

    private float getnumbericalWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        float woe = getnumbericalWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Float> woeMeans = isWeighted ? this.wgtWoeMeanMap : this.woeMeanMap;
        Map<Integer, Float> woeStddevs = isWeighted ? this.wgtWoeStddevMap : this.woeStddevMap;
        float mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        double realCutoff = Normalizer.checkCutOff(this.cutOff);
        return Normalizer.computeZScore(woe, mean, stddev, realCutoff)[0].floatValue();
    }

    private float getnumbericalZScoreValue(Integer columnNum, Object obj) {
        float mean = this.numberMeanMap.get(columnNum);
        float stddev = this.numberStddevMap.get(columnNum);
        float rawValue;
        if(obj == null || obj.toString().length() == 0) {
            rawValue = defaultMissingValue(mean);
        } else {
            try {
                rawValue = Float.parseFloat(obj.toString());
            } catch (Exception e) {
                rawValue = defaultMissingValue(mean);
            }
        }
        double realCutoff = Normalizer.checkCutOff(this.cutOff);
        return Normalizer.computeZScore(rawValue, mean, stddev, realCutoff)[0].floatValue();
    }

    public WideAndDeep getWnd() {
        return wnd;
    }

}
