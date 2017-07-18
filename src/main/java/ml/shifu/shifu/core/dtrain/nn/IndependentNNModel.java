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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.GZIPInputStream;

import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.encog.ml.data.basic.BasicMLData;

/**
 * TODO
 * 
 * @author pengzhang
 */
public class IndependentNNModel {

    /**
     * Encog based neural network instance which is used to compute nn score
     */
    private BasicFloatNetwork basicNetwork;

    /**
     * Normalization type
     */
    private NormType normType;

    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private Map<Integer, String> numNameMappings;

    /**
     * Mapping for (ColumnNum, Category List) for categorical feature
     */
    private Map<Integer, List<String>> categoricalColumnNameNames;

    /**
     * Mapping for (ColumnNum, index in double[] array)
     */
    private Map<Integer, Integer> columnNumIndexMappings;

    /**
     * Mapping for (columnNum, (category, woeValue))
     */
    private Map<Integer, Map<String, Double>> categoricalWoeMappings;

    /**
     * Mapping for (columnNum, (category, weightedWoeValue))
     */
    private Map<Integer, Map<String, Double>> weightedCategoricalWoeMappings;

    /**
     * Mapping for (columnNum, (category, posRate)) for categorical columns
     */
    private Map<Integer, Map<String, Double>> binPosRateMappings;

    /**
     * Mapping for (columnNum, binBoundaries) for numerical columns
     */
    private Map<Integer, List<Double>> numericalBinBoundaries;

    /**
     * Mapping for (columnNum, weightedBinWoes) for numerical columns, last one in weightedBinWoes is for missing value
     * bin
     */
    private Map<Integer, List<Double>> numericalWeightedWoes;

    /**
     * Mapping for (columnNum, weightedWoes) for numerical columns, last one in weightedWoes is for missing value bin
     */
    private Map<Integer, List<Double>> numericalWoes;

    /**
     * ZScore stddev cutoff value
     */
    private double cutOff;

    /**
     * Mapping for (columnNum, mean) for all columns
     */
    private Map<Integer, Double> numericalMeanMappings;

    /**
     * Mapping for (columnNum, stddev) for all columns
     */
    private Map<Integer, Double> numericalStddevMappings;

    /**
     * Mapping for (columnNum, woeMean) for all columns
     */
    private Map<Integer, Double> woeMeanMappings;

    /**
     * Mapping for (columnNum, woeStddev) for all columns
     */
    private Map<Integer, Double> woeStddevMappings;

    /**
     * Mapping for (columnNum, weightedWoeMean) for all columns
     */
    private Map<Integer, Double> weightedWoeMeanMappings;

    /**
     * Mapping for (columnNum, weightedWoeStddev) for all columns
     */
    private Map<Integer, Double> weightedWoeStddevMappings;

    /**
     * Model version
     */
    @SuppressWarnings("unused")
    private static int version = CommonConstants.NN_FORMAT_VERSION;

    public double[] compute(double[] data) {
        return this.basicNetwork.compute(new BasicMLData(data)).getData();
    }

    public double[] compute(Map<String, Object> dataMap) {
        return compute(convertDataMapToDoubleArray(dataMap));
    }

    private double[] convertDataMapToDoubleArray(Map<String, Object> dataMap) {
        double[] data = new double[this.columnNumIndexMappings.size()];
        for(Entry<Integer, Integer> entry: this.columnNumIndexMappings.entrySet()) {
            double value = 0d;
            Integer columnNum = entry.getKey();
            String columnName = this.numNameMappings.get(columnNum);
            Object obj = dataMap.get(columnName);
            if(this.categoricalColumnNameNames.containsKey(columnNum)) {
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
                        value = getCategoricalPosRateValue(columnNum, obj);
                        break;
                }
            } else {
                // numerical column
                switch(this.normType) {
                    case WOE:
                        value = getNumericalWoeValue(columnNum, obj, false);
                        break;
                    case WEIGHT_WOE:
                        value = getNumericalWoeValue(columnNum, obj, false);
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
            }

            Integer index = entry.getValue();
            if(index != null && index < data.length) {
                data[index] = value;
            }
        }
        return data;
    }

    private double getNumericalWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        double value;
        int binIndex = -1;
        if(obj != null) {
            binIndex = CommonUtils.getNumericalBinIndex(this.numericalBinBoundaries.get(columnNum), obj.toString());
        }
        List<Double> binWoes = isWeighted ? this.numericalWeightedWoes.get(columnNum) : this.numericalWoes
                .get(columnNum);
        if(binIndex == -1) {
            // The last bin in woeBins is the miss value bin.
            value = binWoes.get(binWoes.size() - 1);
        } else {
            value = binWoes.get(binIndex);
        }
        return value;
    }

    private double getNumericalZScoreValue(Integer columnNum, Object obj) {
        double rawValue = 0d;
        double mean = this.numericalMeanMappings.get(columnNum);
        if(obj == null || obj.toString().length() == 0) {
            rawValue = defaultMissingValue(mean);
        } else {
            try {
                rawValue = Double.parseDouble(obj.toString());
            } catch (Exception e) {
                rawValue = defaultMissingValue(mean);
            }
        }
        double stddev = this.numericalStddevMappings.get(columnNum);
        return Normalizer.computeZScore(rawValue, mean, stddev, this.cutOff);
    }

    private double getCategoricalPosRateValue(Integer columnNum, Object obj) {
        double value = 0d;
        Map<String, Double> posRateMapping = this.binPosRateMappings.get(columnNum);
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
        return value;
    }

    private double getNumericalWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getNumericalWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.weightedWoeMeanMappings : this.woeMeanMappings;
        Map<Integer, Double> woeStddevs = isWeighted ? this.weightedWoeStddevMappings : this.woeStddevMappings;
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        return Normalizer.computeZScore(woe, mean, stddev, this.cutOff);
    }

    private double getCategoricalWoeZScoreValue(Integer columnNum, Object obj, boolean isWeighted) {
        double woe = getCategoricalWoeValue(columnNum, obj, isWeighted);
        Map<Integer, Double> woeMeans = isWeighted ? this.weightedWoeMeanMappings : this.woeMeanMappings;
        Map<Integer, Double> woeStddevs = isWeighted ? this.weightedWoeStddevMappings : this.woeStddevMappings;
        double mean = woeMeans.get(columnNum), stddev = woeStddevs.get(columnNum);
        return Normalizer.computeZScore(woe, mean, stddev, this.cutOff);
    }

    private double getCategoricalWoeValue(Integer columnNum, Object obj, boolean isWeighted) {
        double value = 0d;
        Map<Integer, Map<String, Double>> mappings = isWeighted ? this.weightedCategoricalWoeMappings
                : categoricalWoeMappings;
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
        int inputCnt = dis.readInt();
        NormType normType = NormType.valueOf(dis.readUTF().toUpperCase());
        
        Map<Integer, String> numNameMap = new HashMap<Integer, String>();
        Map<Integer, List<String>> cateColumnNameNames  = new HashMap<Integer, List<String>>();
        
        Map<Integer, Map<String, Double>> cateWoeMap = new HashMap<Integer, Map<String,Double>>();
        Map<Integer, Map<String, Double>> cateWgtWoeMap = new HashMap<Integer, Map<String,Double>>();
        
        List<NNColumnStats> columnStatsList = new ArrayList<NNColumnStats>();
        int size = dis.readInt();
        for(int i = 0; i < size; i++) {
            NNColumnStats cs = new NNColumnStats();
            cs.readFields(dis);
            
            numNameMap.put(cs.getColumnNum(), cs.getColumnName());
            if(cs.isCategorical()) {
                List<String> binCategories = cs.getBinCategories();
                cateColumnNameNames.put(cs.getColumnNum(), binCategories);
                
                Map<String, Double> woeMap = new HashMap<String, Double>();
                for(int j = 0; j < binCategories.size(); j++) {
                    String currCate = binCategories.get(j);
                    /*if(currCate.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                        // merged category should be flatten, use split function this class to avoid depending on guava jar
                        String[] splits = split(currCate, Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
                        for(String str: splits) {
                            categoryIndexMapping.put(str, j);
                        }
                    } else {
                        categoryIndexMapping.put(category, j);
                    }*/
                }
            } else {
                
            }
            
            columnStatsList.add(cs);
        }
        
        BasicFloatNetwork network = new PersistBasicFloatNetwork().readNetwork(dis);
        return new IndependentNNModel();
    }

}
