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
import ml.shifu.shifu.util.Constants;

/**
 * {@link IndependentTreeModel} depends no other classes which is easy to deploy model in production.
 * 
 * <p>
 * {@link #loadFromStream(InputStream)} should be the only interface to load a tree model object.
 * 
 * <p>
 * To predict data for tree model, call {@link #compute(Map)} or {@link #compute(double[])}
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class IndependentTreeModel {

    private static final String NAMESPACE_DELIMITER = "::";

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
     * A list of tree models, can be RF or GBT
     */
    private List<TreeNode> trees;

    /**
     * Weights per each tree in {@link #trees}
     */
    private List<Double> weights;

    /**
     * If it is for GBT
     */
    private boolean isGBDT = false;

    /**
     * If model is for classification
     */
    private boolean isClassification = false;

    /**
     * GBT model results is not in [0, 1], set {@link #isConvertToProb} to true will normalize model score to [0, 1]
     */
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
    private int version;

    /**
     * For numerical columns, mean value is used for null replacement
     */
    private Map<Integer, Double> numericalMeanMapping;

    public IndependentTreeModel(Map<Integer, Double> numericalMeanMapping, Map<Integer, String> numNameMapping,
            Map<Integer, List<String>> categoricalColumnNameNames,
            Map<Integer, Map<String, Integer>> columnCategoryIndexMapping, Map<Integer, Integer> columnNumIndexMapping,
            boolean isOptimizeMode, List<TreeNode> trees, List<Double> weights, boolean isGBDT,
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
        this.lossStr = lossStr;
        this.algorithm = algorithm;
        this.inputNode = inputNode;
        this.version = version;

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
     * Run gbt model as classification
     * 
     * @param data
     *            - double data map (for numerical variable, it is double value, for categorical variable it is index)
     * @return classification result
     */
    private double[] computeClassificationScore(double[] data) {
        double[] scores = new double[this.trees.size()];
        for(int i = 0; i < this.trees.size(); i++) {
            TreeNode treeNode = this.trees.get(i);
            double score = predictNode(treeNode.getNode(), data);
            scores[i] = score;
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
        double predictSum = 0d;
        double weightSum = 0d;
        for(int i = 0; i < this.trees.size(); i++) {
            TreeNode treeNode = this.trees.get(i);
            Double weight = this.weights.get(i);
            weightSum += weight;
            double score = predictNode(treeNode.getNode(), data);
            predictSum += score * weight;
        }

        double finalPredict;
        if(this.isGBDT) {
            if(this.isConvertToProb) {
                finalPredict = convertToProb(predictSum);
            } else {
                finalPredict = predictSum;
            }
        } else {
            finalPredict = predictSum / weightSum;
        }
        return new double[] { finalPredict };
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
     * Covert score to probability value which are in [0, 1], for GBT regression, scores can not be [0, 1]. Round score
     * to 1.0E19 to avoid NaN in final return result.
     * 
     * @param score
     *            the raw score
     * @return score after sigmoid transform.
     */
    public double convertToProb(double score) {
        // sigmoid function to covert to [0, 1], TODO, how to make it configuable for users
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
                    Integer intIndex = columnCategoryIndexMapping.get(columnNum).get(obj.toString());
                    if(intIndex == null || intIndex < 0 || intIndex >= categoricalSize) {
                        // last one is for invalid category
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
    public List<TreeNode> getTrees() {
        return trees;
    }

    /**
     * @return the weights
     */
    public List<Double> getWeights() {
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
    public void setTrees(List<TreeNode> trees) {
        this.trees = trees;
    }

    /**
     * @param weights
     *            the weights to set
     */
    public void setWeights(List<Double> weights) {
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
                if(columnName.contains(NAMESPACE_DELIMITER)) {
                    columnName = columnName
                            .substring(columnName.indexOf(NAMESPACE_DELIMITER) + NAMESPACE_DELIMITER.length(),
                                    columnName.length());
                }
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
                String category = dis.readUTF();
                // categories is merged category list
                categories.add(category);
                if(category.contains(Constants.CATEGORICAL_GROUP_VAL_DELIMITER)) {
                    // merged category should be flatten, use split function this class to avoid depending on guava jar
                    String[] splits = split(category, Constants.CATEGORICAL_GROUP_VAL_DELIMITER);
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

        // if one vs all, even multiple classification, treated as regression
        return new IndependentTreeModel(numericalMeanMapping, columnIndexNameMapping, categoricalColumnNameNames,
                columnCategoryIndexMapping, columnMapping, isOptimizeMode, trees, weights,
                CommonConstants.GBT_ALG_NAME.equalsIgnoreCase(algorithm), isClassification && !isOneVsAll,
                isConvertToProb, lossStr, algorithm, inputNode, version);
    }

    /**
     * Manual split function to avoid depending on guava.
     * 
     * <p>
     * Some examples: "^"=&gt;[, ]; ""=&gt;[]; "a"=&gt;[a]; "abc"=&gt;[abc]; "a^"=&gt;[a, ]; "^b"=&gt;[, b];
     * "^^b"=&gt;[, , b]
     * 
     * @param str
     *            the string to be split
     * @param delimiter
     *            the delimiter
     * @return split string array
     */
    public static String[] split(String str, String delimiter) {
        if(str == null || str.length() == 0) {
            return new String[] { "" };
        }

        List<String> categories = new ArrayList<String>();
        int dLen = delimiter.length();
        int begin = 0;
        for(int i = 0; i < str.length(); i++) {
            if(str.substring(i, Math.min(i + dLen, str.length())).equals(delimiter)) {
                categories.add(str.substring(begin, i));
                begin = i + dLen;
            }
            if(i == str.length() - 1) {
                categories.add(str.substring(begin, str.length()));
            }
        }

        return categories.toArray(new String[0]);
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

    /**
     * @return the version
     */
    public int getVersion() {
        return version;
    }

    /**
     * @param version
     *            the version to set
     */
    public void setVersion(int version) {
        this.version = version;
    }

}
