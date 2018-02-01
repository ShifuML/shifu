/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.dt;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UTFDataFormatException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.zip.GZIPInputStream;

import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.util.Constants;

/**
 * {@link IndependentTreeModel} depends no other classes which is easy to deploy model in production.
 * 
 * <p>
 * {@link #loadFromStream(InputStream)} should be the only interface to load a tree model object.
 * 
 * <p>
 * To predict data for tree model, call {@link #compute(Map)} or {@link #compute(double[])}
 */
public class IndependentTreeModel {

    /**
     * Mapping for (ColumnNum, ColumnName)
     */
    private Map<Integer, String> numNameMapping;

    /**
     * Mapping for (ColumnNum, Category List) for categorical feature
     */
    private Map<Integer, List<String>> categoricalColumnNameNames;

    /**
     * Mapping for (ColumnNum, Map(Category, CategoryIndex) for categorical feature
     */
    private Map<Integer, Map<String, Integer>> columnCategoryIndexMapping;

    /**
     * indicator for the model is loaded as optimize mode
     */
    private boolean isOptimizeMode = false;

    /**
     * Caching the value size for categorical variables to avoid map query
     */
    private int[] categoricalValueSize;

    /**
     * Mapping for (ColumnNum, index in double[] array)
     */
    private Map<Integer, Integer> columnNumIndexMapping;

    /**
     * A list of tree ensemble models, can be RF or GBT, starting from 0.11.0, change it to ist<List<TreeNode>> to
     * support bagging models.
     */
    private List<List<TreeNode>> trees;

    /**
     * Weights per each tree in {@link #trees}
     */
    private List<List<Double>> weights;

    /**
     * If it is for GBT
     */
    private boolean isGBDT = false;

    /**
     * If model is for classification
     */
    private boolean isClassification = false;

    /**
     * GBT model scores are not in [0, 1], set {@link #isConvertToProb} to true will normalize model score to [0, 1].
     * 
     * <p>
     * Deprecate this field and make more strategies in {@link #gbtScoreConvertStrategy}, RAW and SIGMOID are the same
     * as isConvertToProb to false or true. If {@link #gbtScoreConvertStrategy} is null, #isConvertToProb still is
     * effective.
     */
    @Deprecated
    private boolean isConvertToProb = false;

    /**
     * {@link #lossStr} is used to validate, if continuous model training but different loss type, should be failed.
     * TODO add validation
     */
    private String lossStr;

    /**
     * RF or GBT
     */
    private String algorithm;

    /**
     * # of input node
     */
    private int inputNode;

    /**
     * Model version
     */
    private static int version = CommonConstants.TREE_FORMAT_VERSION;

    /**
     * For numerical columns, mean value is used for null replacement
     */
    private Map<Integer, Double> numericalMeanMapping;

    /**
     * GBT model scores are not in [0, 1], to make it in [0, 1], different strategies can be provided. Set this field as
     * String not Enum to avoid dependency on json related jars.
     */
    private String gbtScoreConvertStrategy = Constants.GBT_SCORE_RAW_CONVETER;

    /**
     * IF current scoring is for gbt sigmoid convert, a flag here to avoid multiple string comparsion, for old sigmoid
     * transform, which is code without scale: <code>1 / (1 + Math.min(1.0E19, Math.exp(- score))); </code>.
     */
    private boolean isGBTOldSigmoidConvert;

    /**
     * IF current scoring is for gbt sigmoid convert, a flag here to avoid multiple string comparsion, for new sigmoid
     * transform, which is code without scale: <code>1 / (1 + Math.min(1.0E19, Math.exp(-20 * score))); </code>.
     */
    private boolean isGBTSigmoidConvert;

    /**
     * IF current scoring is for gbt cut off convert, a flag here to avoid multiple string comparsion
     */
    private boolean isGBTCutoffConvert;

    /**
     * IF current scoring is raw gbt score, a flag here to avoid multiple string comparsion
     */
    @SuppressWarnings("unused")
    private boolean isGBTRawScore;

    public IndependentTreeModel(Map<Integer, Double> numericalMeanMapping, Map<Integer, String> numNameMapping,
            Map<Integer, List<String>> categoricalColumnNameNames,
            Map<Integer, Map<String, Integer>> columnCategoryIndexMapping, Map<Integer, Integer> columnNumIndexMapping,
            boolean isOptimizeMode, List<List<TreeNode>> trees, List<List<Double>> weights, boolean isGBDT,
            boolean isClassification, boolean isConvertToProb, String lossStr, String algorithm, int inputNode,
            int version) {
        this.numericalMeanMapping = numericalMeanMapping;
        this.numNameMapping = numNameMapping;
        this.categoricalColumnNameNames = categoricalColumnNameNames;
        this.columnCategoryIndexMapping = columnCategoryIndexMapping;
        this.columnNumIndexMapping = columnNumIndexMapping;
        this.isOptimizeMode = isOptimizeMode;
        this.trees = trees;
        this.weights = weights;
        this.isGBDT = isGBDT;
        this.isClassification = isClassification;
        this.isConvertToProb = isConvertToProb;

        if(this.isConvertToProb) {
            this.gbtScoreConvertStrategy = Constants.GBT_SCORE_OLD_SIGMOID_CONVETER;
        } else {
            this.gbtScoreConvertStrategy = Constants.GBT_SCORE_RAW_CONVETER;
        }

        this.lossStr = lossStr;
        this.algorithm = algorithm;
        this.inputNode = inputNode;
        IndependentTreeModel.version = version;

        if(this.isOptimizeMode) {
            // caching value size of categorical variable
            // but just only cache those used categorical variables
            this.categoricalValueSize = new int[this.columnNumIndexMapping.size()];
            Iterator<Entry<Integer, List<String>>> iterator = this.categoricalColumnNameNames.entrySet().iterator();
            while(iterator.hasNext()) {
                Entry<Integer, List<String>> entry = iterator.next();
                Integer columnNum = entry.getKey();
                if(this.columnNumIndexMapping.containsKey(columnNum)) {
                    this.categoricalValueSize[this.columnNumIndexMapping.get(columnNum)] = entry.getValue().size();
                }
            }
        }

        isGBTSigmoidConvert = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_SIGMOID_CONVETER);
        isGBTOldSigmoidConvert = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_OLD_SIGMOID_CONVETER);
        isGBTCutoffConvert = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_CUTOFF_CONVETER);
        isGBTRawScore = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_RAW_CONVETER);
    }

    public IndependentTreeModel(Map<Integer, Double> numericalMeanMapping, Map<Integer, String> numNameMapping,
            Map<Integer, List<String>> categoricalColumnNameNames,
            Map<Integer, Map<String, Integer>> columnCategoryIndexMapping, Map<Integer, Integer> columnNumIndexMapping,
            boolean isOptimizeMode, List<List<TreeNode>> trees, List<List<Double>> weights, boolean isGBDT,
            boolean isClassification, boolean isConvertToProb, String lossStr, String algorithm, int inputNode,
            int version, String gbtScoreConvertStrategy) {
        this(numericalMeanMapping, numNameMapping, categoricalColumnNameNames, columnCategoryIndexMapping,
                columnNumIndexMapping, isOptimizeMode, trees, weights, isGBDT, isClassification, isConvertToProb,
                lossStr, algorithm, inputNode, version);

        if(isValidGbtScoreConvertStrategy(gbtScoreConvertStrategy)) {
            if(isGbtHalfCutOffOrMaxMinStrategy(gbtScoreConvertStrategy)) {
                System.err
                        .println("WARN: For HALF_CUTOFF and MAXMIN_SCALE gbt score scale strategy not supported in IndependentTreeModel and will be treated in RAW, in client side, such score can be rescaled by return score values.");
                this.gbtScoreConvertStrategy = Constants.GBT_SCORE_RAW_CONVETER;
            } else {
                this.gbtScoreConvertStrategy = gbtScoreConvertStrategy;
            }
        } else {
            // set gbtScoreConvertStrategy by isConvertToProb in the other constructor
        }

        // re-check boolean flags compute for fast check in compute method
        isGBTSigmoidConvert = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_SIGMOID_CONVETER);
        isGBTOldSigmoidConvert = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_OLD_SIGMOID_CONVETER);
        isGBTCutoffConvert = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_CUTOFF_CONVETER);
        isGBTRawScore = this.gbtScoreConvertStrategy != null
                && this.gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_RAW_CONVETER);
    }

    private static boolean isGbtHalfCutOffOrMaxMinStrategy(String gbtScoreConvertStrategy) {
        return gbtScoreConvertStrategy != null
                && (gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_HALF_CUTOFF_CONVETER) || gbtScoreConvertStrategy
                        .equalsIgnoreCase(Constants.GBT_SCORE_MAXMIN_SCALE_CONVETER));
    }

    public static boolean isValidGbtScoreConvertStrategy(String gbtScoreConvertStrategy) {
        return gbtScoreConvertStrategy != null
                && (gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_RAW_CONVETER)
                        || gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_SIGMOID_CONVETER)
                        || gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_OLD_SIGMOID_CONVETER)
                        || gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_CUTOFF_CONVETER)
                        || gbtScoreConvertStrategy.equalsIgnoreCase(Constants.GBT_SCORE_HALF_CUTOFF_CONVETER) || gbtScoreConvertStrategy
                            .equalsIgnoreCase(Constants.GBT_SCORE_MAXMIN_SCALE_CONVETER));
    }

    /**
     * Given double array data, compute score values of tree model.
     * 
     * @param data
     *            data array includes only effective column data, numeric value is real value, categorical feature value
     *            is index of binCategoryList.
     * @return if classification mode, return array of all scores of trees
     *         if regression of RF, return array with only one element which is avg score of all tree model scores
     *         if regression of GBT, return array with only one element which is score of the GBT model
     */
    public double[] compute(double[] data) {
        return (this.isClassification ? computeClassificationScore(data) : computeRegressionScore(data));
    }

    /**
     * Run as classification mode, since no idea of average or vote, classification will return all tree values.
     * 
     * @param data
     *            - double data map (for numerical variable, it is double value, for categorical variable it is index)
     * @return classification result
     */
    private double[] computeClassificationScore(double[] data) {
        List<Double> scoreList = new ArrayList<Double>();
        for(int i = 0; i < this.trees.size(); i++) {
            List<TreeNode> list = this.trees.get(i);
            for(int j = 0; j < list.size(); j++) {
                TreeNode treeNode = list.get(j);
                scoreList.add(predictNode(treeNode.getNode(), data));
            }
        }

        double[] scores = new double[scoreList.size()];
        for(int i = 0; i < scores.length; i++) {
            scores[i] = scoreList.get(i);
        }
        return scores;
    }

    /**
     * Run gbt model as regression
     * 
     * @param data
     *            - double data map (for numerical variable, it is double value, for categorical variable it is index)
     * @return regression result
     */
    private double[] computeRegressionScore(double[] data) {
        if(this.isGBDT) {
            // GBDT prediction
            int bags = this.trees.size();
            double finalPredict = 0d;
            for(int i = 0; i < bags; i++) {
                // compute one gbt model score
                List<TreeNode> list = this.trees.get(i);
                List<Double> wgtList = this.weights.get(i);
                double predict = 0d;
                for(int j = 0; j < list.size(); j++) {
                    TreeNode treeNode = list.get(j);
                    double score = predictNode(treeNode.getNode(), data);
                    predict += score * wgtList.get(j);
                }

                if(this.isGBTOldSigmoidConvert) {
                    predict = convertToSigmoid(predict);
                } else if(this.isGBTSigmoidConvert) {
                    predict = convertToNewSigmoid(predict);
                } else if(this.isGBTCutoffConvert) {
                    predict = cutoffPredict(predict);
                } else {
                    // raw score, not to do sth. just to use raw predict score
                    // here, only SIGMOID, RAW, CUTOFF are supported here since for HALF_CUTOFF and MAXMIN_SCALE are all
                    // related with max and min score and has been processed in ConfusionMatrix, for HALF_CUTOFF or
                    // MAXMIN_SCALE, just keep it as raw score here without doing anything
                }

                // sum all computing scores
                finalPredict += predict;
            }
            // return average bagging score in
            return new double[] { finalPredict / bags };
        } else {
            // RF prediction
            int bags = this.trees.size();
            double finalPredict = 0d;
            for(int i = 0; i < bags; i++) {
                // compute one RF model score
                List<TreeNode> list = this.trees.get(i);
                List<Double> wgtList = this.weights.get(i);
                double predictSum = 0d, weightSum = 0d;
                for(int j = 0; j < list.size(); j++) {
                    TreeNode treeNode = list.get(j);
                    double score = predictNode(treeNode.getNode(), data);
                    double weight = wgtList.get(j);
                    weightSum += weight;
                    predictSum += score * weight;
                }

                // sum all computing scores (score is current RF score)
                finalPredict += (predictSum / weightSum);
            }
            // return average bagging score in all RFs
            return new double[] { finalPredict / bags };
        }

    }

    /**
     * Given {@code dataMap} with format (columnName, value), compute score values of tree model.
     * 
     * <p>
     * No any alert or exception if your {@code dataMap} doesn't contain features included in the model, such case will
     * be treated as missing value case. Please make sure feature names in keys of {@code dataMap} are consistent with
     * names in model.
     * 
     * <p>
     * In {@code dataMap}, numerical value can be (String, Double) format or (String, String) format, they will all be
     * parsed to Double; categorical value are all converted to (String, String). If value not in our categorical list,
     * it will also be treated missing value.
     * 
     * @param dataMap
     *            {@code dataMap} for (columnName, value), numeric value can be double/String, categorical feature can
     *            be int(index) or category value. if not set or set to null, such feature will be treated as missing
     *            value. For numerical value, if it cannot be parsed successfully, it will also be treated as missing.
     * @return if classification mode, return array of all scores of trees
     *         if regression of RF, return array with only one element which is average score of all tree model scores
     *         if regression of GBT, return array with only one element which is score of the GBT model
     */
    public final double[] compute(Map<String, Object> dataMap) {
        return compute(convertDataMapToDoubleArray(dataMap));
    }

    /**
     * Cut off predict score in [0, 1] range since for binary regression usually score smaller than 0 can be treated as
     * 0 and score larger than 1 can be treated as 1
     * 
     * @param score
     *            the raw score
     * @return cut off score which should be in [0, 1]
     */
    public double cutoffPredict(double score) {
        if(score < 0d) {
            return 0d;
        } else if(score > 1d) {
            return 1d;
        } else {
            return score;
        }
    }

    /**
     * Covert score to probability value which are in [0, 1], for GBT regression, scores can not be [0, 1]. Round score
     * to 1.0E19 to avoid NaN in final return result. Sigmoid function is new one:
     * <code>1 / (1 + Math.min(1.0E19, Math.exp(- score))); </code>.
     * 
     * @param score
     *            the raw score
     * @return score after sigmoid transform.
     */
    public double convertToNewSigmoid(double score) {
        return 1 / (1 + Math.min(1.0E19, Math.exp(-20 * score)));
    }

    /**
     * Covert score to probability value which are in [0, 1], for GBT regression, scores can not be [0, 1]. Round score
     * to 1.0E19 to avoid NaN in final return result. Sigmoid function is old one for compatablity:
     * <code>1 / (1 + Math.min(1.0E19, Math.exp(- 20 * score))); </code>.
     * 
     * @param score
     *            the raw score
     * @return score after sigmoid transform.
     */
    public double convertToSigmoid(double score) {
        return 1 / (1 + Math.min(1.0E19, Math.exp(-score)));
    }

    private double predictNode(Node topNode, double[] data) {
        Node currNode = topNode;
        // go until leaf
        while(currNode.getSplit() != null && !currNode.isRealLeaf()) {
            Split split = currNode.getSplit();
            double value = data[this.getColumnIndex(split.getColumnNum())];
            if(split.getFeatureType() == Split.CONTINUOUS) {
                // value is real numeric value and no need to transform to binLowestValue
                if(value < split.getThreshold()) {
                    currNode = currNode.getLeft();
                } else {
                    currNode = currNode.getRight();
                }
            } else if(split.getFeatureType() == Split.CATEGORICAL) {
                short indexValue = -1;
                int categoricalSize = this.getCategoricalSize(split.getColumnNum());
                if(Double.compare(value, 0d) < 0 || Double.compare(value, categoricalSize) >= 0) {
                    indexValue = (short) categoricalSize;
                } else {
                    // value is category index + 0.1d is to avoid 0.9999999 converted to 0, is there?
                    indexValue = (short) (value + 0.1d);
                }
                Set<Short> childCategories = split.getLeftOrRightCategories();
                if(split.isLeft()) {
                    if(childCategories.contains(indexValue)) {
                        currNode = currNode.getLeft();
                    } else {
                        currNode = currNode.getRight();
                    }
                } else {
                    if(childCategories.contains(indexValue)) {
                        currNode = currNode.getRight();
                    } else {
                        currNode = currNode.getLeft();
                    }
                }
            }
        }

        if(this.isClassification) {
            return currNode.getPredict().getClassValue();
        } else {
            return currNode.getPredict().getPredict();
        }
    }

    private int getColumnIndex(int columnNum) {
        return (this.isOptimizeMode ? columnNum : this.columnNumIndexMapping.get(columnNum));
    }

    private int getCategoricalSize(int columnNum) {
        return (this.isOptimizeMode ? this.categoricalValueSize[columnNum] : categoricalColumnNameNames.get(columnNum)
                .size());
    }

    private double[] convertDataMapToDoubleArray(Map<String, Object> dataMap) {
        double[] data = new double[this.columnNumIndexMapping.size()];
        for(Entry<Integer, Integer> entry: this.columnNumIndexMapping.entrySet()) {
            double value = 0d;
            Integer columnNum = entry.getKey();
            String columnName = this.numNameMapping.get(columnNum);
            Object obj = dataMap.get(columnName);
            if(this.categoricalColumnNameNames.containsKey(columnNum)) {
                // categorical column
                double indexValue = -1d;
                int categoricalSize = categoricalColumnNameNames.get(columnNum).size();
                if(obj == null) {
                    // no matter set it to null or not set it in dataMap, it will be treated as missing value, last one
                    // is missing value category
                    indexValue = categoricalSize;
                } else {
                    Map<String, Integer> categoryIndexMap = columnCategoryIndexMapping.get(columnNum);
                    Integer intIndex = categoryIndexMap.get(obj.toString());
                    if(intIndex == null || intIndex < 0 || intIndex >= categoricalSize) {
                        // cannot find category, set it to missing bin (last one)
                        intIndex = categoricalSize;
                    }
                    indexValue = intIndex;
                }
                value = indexValue;
            } else {
                // numerical column
                if(obj == null || ((obj instanceof String) && ((String) obj).length() == 0)) {
                    // no matter set it to null or not set it in dataMap, it will be treated as missing value, last one
                    // is missing value category
                    value = this.numericalMeanMapping.get(columnNum) == null ? 0d : this.numericalMeanMapping
                            .get(columnNum);
                } else {
                    if(obj instanceof Number) {
                        value = ((Number) obj).doubleValue();
                    } else {
                        try {
                            value = Double.parseDouble(obj.toString());
                        } catch (NumberFormatException e) {
                            // not valid double value for numerical feature, using default value
                            value = this.numericalMeanMapping.get(columnNum) == null ? 0d : this.numericalMeanMapping
                                    .get(columnNum);
                        }
                    }
                }
                if(Double.isNaN(value)) {
                    value = this.numericalMeanMapping.get(columnNum) == null ? 0d : this.numericalMeanMapping
                            .get(columnNum);
                }
            }
            Integer index = entry.getValue();
            if(index != null && index < data.length) {
                data[index] = value;
            }
        }
        return data;
    }

    /**
     * @return the lossStr
     */
    public String getLossStr() {
        return lossStr;
    }

    /**
     * @param lossStr
     *            the lossStr to set
     */
    public void setLossStr(String lossStr) {
        this.lossStr = lossStr;
    }

    /**
     * @return the numNameMapping
     */
    public Map<Integer, String> getNumNameMapping() {
        return numNameMapping;
    }

    /**
     * @return the categoricalColumnNameNames
     */
    public Map<Integer, List<String>> getCategoricalColumnNameNames() {
        return categoricalColumnNameNames;
    }

    /**
     * @return the columnCategoryIndexMapping
     */
    public Map<Integer, Map<String, Integer>> getColumnCategoryIndexMapping() {
        return columnCategoryIndexMapping;
    }

    /**
     * @return the columnNumIndexMapping
     */
    public Map<Integer, Integer> getColumnNumIndexMapping() {
        return columnNumIndexMapping;
    }

    /**
     * @return the trees
     */
    public List<List<TreeNode>> getTrees() {
        return trees;
    }

    /**
     * @return the weights
     */
    public List<List<Double>> getWeights() {
        return weights;
    }

    /**
     * @return the isGBDT
     */
    public boolean isGBDT() {
        return isGBDT;
    }

    /**
     * @return the isClassification
     */
    public boolean isClassification() {
        return isClassification;
    }

    /**
     * @return the isConvertToProb
     */
    public boolean isConvertToProb() {
        return isConvertToProb;
    }

    /**
     * @param numNameMapping
     *            the numNameMapping to set
     */
    public void setNumNameMapping(Map<Integer, String> numNameMapping) {
        this.numNameMapping = numNameMapping;
    }

    /**
     * @param categoricalColumnNameNames
     *            the categoricalColumnNameNames to set
     */
    public void setCategoricalColumnNameNames(Map<Integer, List<String>> categoricalColumnNameNames) {
        this.categoricalColumnNameNames = categoricalColumnNameNames;
    }

    /**
     * @param columnCategoryIndexMapping
     *            the columnCategoryIndexMapping to set
     */
    public void setColumnCategoryIndexMapping(Map<Integer, Map<String, Integer>> columnCategoryIndexMapping) {
        this.columnCategoryIndexMapping = columnCategoryIndexMapping;
    }

    /**
     * @param columnNumIndexMapping
     *            the columnNumIndexMapping to set
     */
    public void setColumnNumIndexMapping(Map<Integer, Integer> columnNumIndexMapping) {
        this.columnNumIndexMapping = columnNumIndexMapping;
    }

    /**
     * @param trees
     *            the trees to set
     */
    public void setTrees(List<List<TreeNode>> trees) {
        this.trees = trees;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(List<List<Double>> weights) {
        this.weights = weights;
    }

    /**
     * @param isGBDT
     *            the isGBDT to set
     */
    public void setGBDT(boolean isGBDT) {
        this.isGBDT = isGBDT;
    }

    /**
     * @param isClassification
     *            the isClassification to set
     */
    public void setClassification(boolean isClassification) {
        this.isClassification = isClassification;
    }

    /**
     * @param isConvertToProb
     *            the isConvertToProb to set
     */
    public void setConvertToProb(boolean isConvertToProb) {
        this.isConvertToProb = isConvertToProb;
    }

    /**
     * @return the algorithm
     */
    public String getAlgorithm() {
        return algorithm;
    }

    /**
     * @param algorithm
     *            the algorithm to set
     */
    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }

    /**
     * @return the inputNode
     */
    public int getInputNode() {
        return inputNode;
    }

    /**
     * @param inputNode
     *            the inputNode to set
     */
    public void setInputNode(int inputNode) {
        this.inputNode = inputNode;
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf, by default not to convert gbt score to [0, 1]
     * 
     * @param input
     *            the input stream
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, false);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf, by default not to convert gbt score to [0, 1]
     * 
     * @param input
     *            the input stream
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, String gbtScoreConvertStrategy)
            throws IOException {
        return loadFromStream(input, false, false, true, gbtScoreConvertStrategy);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf. User can specify to use raw score or score after
     * sigmoid transform by isConvertToProb.
     * 
     * @param input
     *            the input stream
     * @param isConvertToProb
     *            if convert score to probability (if to transform raw score by sigmoid)
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, boolean isConvertToProb) throws IOException {
        return loadFromStream(input, isConvertToProb, false, true);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf. User can specify to use raw score or score after
     * sigmoid transform by isConvertToProb.
     * 
     * @param input
     *            the input stream
     * @param isConvertToProb
     *            if convert score to probability (if to transform raw score by sigmoid)
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, boolean isConvertToProb,
            String gbtScoreConvertStrategy) throws IOException {
        return loadFromStream(input, isConvertToProb, false, true, gbtScoreConvertStrategy);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf. User can specify to use raw score or score after
     * sigmoid transform by isConvertToProb.
     * 
     * @param input
     *            the input stream
     * @param isConvertToProb
     *            if convert score to probability (if to transform raw score by sigmoid)
     * @param isOptimizeMode
     *            if column index query is optimized
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, boolean isConvertToProb, boolean isOptimizeMode)
            throws IOException {
        return loadFromStream(input, isConvertToProb, isOptimizeMode, true);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf. User can specify to use raw score or score after
     * sigmoid transform by isConvertToProb.
     * 
     * @param input
     *            the input stream
     * @param isConvertToProb
     *            if convert score to probability (if to transform raw score by sigmoid)
     * @param isOptimizeMode
     *            if column index query is optimized
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, boolean isConvertToProb,
            boolean isOptimizeMode, String gbtScoreConvertStrategy) throws IOException {
        return loadFromStream(input, isConvertToProb, isOptimizeMode, true, gbtScoreConvertStrategy);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf. User can specify to use raw score or score after
     * sigmoid transfrom by isConvertToProb.
     * 
     * @param input
     *            the input stream
     * @param isConvertToProb
     *            if convert score to probability (if to transform raw score by sigmoid)
     * @param isOptimizeMode
     *            if column index query is optimized
     * @param isRemoveNameSpace
     *            new column name including namespace like "a::b", if true, remove "a::" and set column name to simple
     *            name
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, boolean isConvertToProb,
            boolean isOptimizeMode, boolean isRemoveNameSpace) throws IOException {
        return loadFromStream(input, isConvertToProb, isOptimizeMode, isRemoveNameSpace,
                isConvertToProb ? Constants.GBT_SCORE_OLD_SIGMOID_CONVETER : Constants.GBT_SCORE_RAW_CONVETER);
    }

    /**
     * Load model instance from stream like model0.gbt or model0.rf. User can specify to use raw score or score after
     * sigmoid transform by isConvertToProb.
     * 
     * @param input
     *            the input stream
     * @param isConvertToProb
     *            if convert score to probability (if to transform raw score by sigmoid)
     * @param isOptimizeMode
     *            if column index query is optimized
     * @param isRemoveNameSpace
     *            new column name including namespace like "a::b", if true, remove "a::" and set column name to simple
     *            name
     * @param gbtScoreConvertStrategy
     *            specify how to convert gbt raw score
     * @return the tree model instance
     * @throws IOException
     *             any exception in load input stream
     */
    public static IndependentTreeModel loadFromStream(InputStream input, boolean isConvertToProb,
            boolean isOptimizeMode, boolean isRemoveNameSpace, String gbtScoreConvertStrategy) throws IOException {
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
        IndependentTreeModel.setVersion(version);
        String algorithm = dis.readUTF();
        String lossStr = dis.readUTF();
        boolean isClassification = dis.readBoolean();
        boolean isOneVsAll = dis.readBoolean();
        int inputNode = dis.readInt();

        int size = dis.readInt();
        Map<Integer, Double> numericalMeanMapping = new HashMap<Integer, Double>(size, 1f);
        Map<Integer, String> columnIndexNameMapping = new HashMap<Integer, String>(size, 1f);
        for(int i = 0; i < size; i++) {
            int columnIndex = dis.readInt();
            double mean = dis.readDouble();
            numericalMeanMapping.put(columnIndex, mean);
        }
        size = dis.readInt();
        for(int i = 0; i < size; i++) {
            int columnIndex = dis.readInt();
            String columnName = dis.readUTF();
            if(isRemoveNameSpace) {
                // remove name-space in column name to make it be called by simple name
                columnName = StringUtils.getSimpleColumnName(columnName);
            }
            columnIndexNameMapping.put(columnIndex, columnName);
        }

        size = dis.readInt();
        Map<Integer, List<String>> categoricalColumnNameNames = new HashMap<Integer, List<String>>(size, 1f);
        Map<Integer, Map<String, Integer>> columnCategoryIndexMapping = new HashMap<Integer, Map<String, Integer>>(
                size, 1f);
        for(int i = 0; i < size; i++) {
            int columnIndex = dis.readInt();
            int categoryListSize = dis.readInt();
            Map<String, Integer> categoryIndexMapping = new HashMap<String, Integer>(categoryListSize, 1f);
            List<String> categories = new ArrayList<String>(categoryListSize);
            for(int j = 0; j < categoryListSize; j++) {
                String category = readCategory(dis);
                // categories is merged category list
                categories.add(category);
                if(category.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                    // merged category should be flatten, use split function this class to avoid depending on guava jar
                    String[] splits = StringUtils.split(category, Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
                    for(String str: splits) {
                        categoryIndexMapping.put(str, j);
                    }
                } else {
                    categoryIndexMapping.put(category, j);
                }
            }
            categoricalColumnNameNames.put(columnIndex, categories);
            columnCategoryIndexMapping.put(columnIndex, categoryIndexMapping);
        }

        int columnMappingSize = dis.readInt();
        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnMappingSize, 1f);
        for(int i = 0; i < columnMappingSize; i++) {
            columnMapping.put(dis.readInt(), dis.readInt());
        }

        // for back-forward compatibility, still need to read two floats here for wgtCntRatio
        List<List<TreeNode>> bagTrees = new ArrayList<List<TreeNode>>(1);
        List<List<Double>> bagWgts = new ArrayList<List<Double>>();
        int bags = 0;
        if(IndependentTreeModel.getVersion() < 4) {
            bags = 1;
        } else {
            // if version >=4, model saving first is size
            bags = dis.readInt();
        }
        for(int j = 0; j < bags; j++) {
            int treeNum = dis.readInt();
            List<TreeNode> trees = new CopyOnWriteArrayList<TreeNode>();
            List<Double> weights = new ArrayList<Double>(treeNum);
            for(int i = 0; i < treeNum; i++) {
                TreeNode treeNode = new TreeNode();
                treeNode.readFields(dis);
                trees.add(treeNode);
                weights.add(treeNode.getLearningRate());

                if(isOptimizeMode) {
                    // remap the column number into array index for each node
                    treeNode.remapColumnNum(columnMapping);
                }
            }
            bagTrees.add(trees);
            bagWgts.add(weights);
        }

        // if one vs all, even multiple classification, treated as regression
        return new IndependentTreeModel(numericalMeanMapping, columnIndexNameMapping, categoricalColumnNameNames,
                columnCategoryIndexMapping, columnMapping, isOptimizeMode, bagTrees, bagWgts,
                CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(algorithm), isClassification && !isOneVsAll,
                isConvertToProb, lossStr, algorithm, inputNode, version, gbtScoreConvertStrategy);
    }

    /**
     * Read category by marker, if marker<-1, read from bytes or read from readUTF
     * 
     * @param dis
     *            input stream
     * @return catregory read from input stream
     * @throws IOException
     *             any io exception
     * @throws UnsupportedEncodingException
     *             not supported encoding exception
     */
    private static String readCategory(DataInputStream dis) throws IOException, UnsupportedEncodingException {
        String category = null;
        short markerOrLen = dis.readShort();
        if(markerOrLen < 0) {
            int len = dis.readInt();
            byte[] bytes = new byte[len];
            for(int k = 0; k < bytes.length; k++) {
                bytes[k] = dis.readByte();
            }
            category = new String(bytes, "UTF-8");
        } else {
            category = readUTF(dis, markerOrLen);
        }
        return category;
    }

    /**
     * Copied from DataInputStream since we need't read utflen in the beginning of the method.
     * 
     * @param in
     *            input stream
     * @param utflen
     *            len of utf string
     * @return the string read from stream
     * @throws IOException
     *             any io exception
     */
    private final static String readUTF(DataInputStream in, int utflen) throws IOException {
        byte[] bytearr = null;
        char[] chararr = null;
        bytearr = new byte[utflen];
        chararr = new char[utflen];

        int c, char2, char3;
        int count = 0;
        int chararr_count = 0;

        in.readFully(bytearr, 0, utflen);

        while(count < utflen) {
            c = (int) bytearr[count] & 0xff;
            if(c > 127)
                break;
            count++;
            chararr[chararr_count++] = (char) c;
        }

        while(count < utflen) {
            c = (int) bytearr[count] & 0xff;
            switch(c >> 4) {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                    /* 0xxxxxxx */
                    count++;
                    chararr[chararr_count++] = (char) c;
                    break;
                case 12:
                case 13:
                    /* 110x xxxx 10xx xxxx */
                    count += 2;
                    if(count > utflen)
                        throw new UTFDataFormatException("malformed input: partial character at end");
                    char2 = (int) bytearr[count - 1];
                    if((char2 & 0xC0) != 0x80)
                        throw new UTFDataFormatException("malformed input around byte " + count);
                    chararr[chararr_count++] = (char) (((c & 0x1F) << 6) | (char2 & 0x3F));
                    break;
                case 14:
                    /* 1110 xxxx 10xx xxxx 10xx xxxx */
                    count += 3;
                    if(count > utflen)
                        throw new UTFDataFormatException("malformed input: partial character at end");
                    char2 = (int) bytearr[count - 2];
                    char3 = (int) bytearr[count - 1];
                    if(((char2 & 0xC0) != 0x80) || ((char3 & 0xC0) != 0x80))
                        throw new UTFDataFormatException("malformed input around byte " + (count - 1));
                    chararr[chararr_count++] = (char) (((c & 0x0F) << 12) | ((char2 & 0x3F) << 6) | ((char3 & 0x3F) << 0));
                    break;
                default:
                    /* 10xx xxxx, 1111 xxxx */
                    throw new UTFDataFormatException("malformed input around byte " + count);
            }
        }
        // The number of chars produced may be less than utflen
        return new String(chararr, 0, chararr_count);
    }

    /**
     * @return the numericalMeanMapping
     */
    public Map<Integer, Double> getNumericalMeanMapping() {
        return numericalMeanMapping;
    }

    /**
     * @param numericalMeanMapping
     *            the numericalMeanMapping to set
     */
    public void setNumericalMeanMapping(Map<Integer, Double> numericalMeanMapping) {
        this.numericalMeanMapping = numericalMeanMapping;
    }

    public static int getVersion() {
        return version;
    }

    public static void setVersion(int from) {
        version = from;
    }

    /**
     * @return the gbtScoreConvertStrategy
     */
    public String getGbtScoreConvertStrategy() {
        return gbtScoreConvertStrategy;
    }

}
