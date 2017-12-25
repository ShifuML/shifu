/*
 * Copyright [2013-2015] PayPal Software Foundation
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

import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;

import org.apache.commons.lang3.tuple.MutablePair;
import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

/**
 * {@link TreeModel} is to load Random Forest or Gradient Boosted Decision Tree models from Encog interfaces. If user
 * wouldn't like to depend on encog, {@link IndependentTreeModel} can be used to execute tree model from Shifu.
 * 
 * <p>
 * {@link #loadFromStream(InputStream, boolean)} can be used to read serialized models. Which is delegated to
 * {@link IndependentTreeModel}.
 */
public class TreeModel extends BasicML implements MLRegression {

    private static final long serialVersionUID = 479043597958785224L;

    /**
     * Tree model instance without dependency on encog.
     */
    private transient IndependentTreeModel independentTreeModel;

    /**
     * Constructor on current {@link IndependentTreeModel}
     * 
     * @param independentTreeModel
     *            the independent tree model
     */
    public TreeModel(IndependentTreeModel independentTreeModel) {
        this.independentTreeModel = independentTreeModel;
    }

    /**
     * Compute model score based on given input double array.
     */
    @Override
    public final MLData compute(final MLData input) {
        double[] data = input.getData();
        return new BasicMLData(this.getIndependentTreeModel().compute(data));
    }

    /**
     * How many input columns.
     */
    @Override
    public int getInputCount() {
        return this.getIndependentTreeModel().getInputNode();
    }

    @Override
    public void updateProperties() {
        // No need implementation
    }

    public static TreeModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, false);
    }

    public static TreeModel loadFromStream(InputStream input, String gbtScoreConvertStrategy) throws IOException {
        return loadFromStream(input, false, gbtScoreConvertStrategy);
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb) throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb));
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb, String gbtScoreConvertStrategy)
            throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb, gbtScoreConvertStrategy));
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb, boolean isOptimizeMode)
            throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb, isOptimizeMode));
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb, boolean isOptimizeMode,
            String gbtScoreConvertStrategy) throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb, isOptimizeMode,
                gbtScoreConvertStrategy));
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb, boolean isOptimizeMode,
            boolean isRemoveNameSpace) throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb, isOptimizeMode,
                isRemoveNameSpace));
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb, boolean isOptimizeMode,
            boolean isRemoveNameSpace, String gbtScoreConvertStrategy) throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb, isOptimizeMode,
                isRemoveNameSpace, gbtScoreConvertStrategy));
    }

    @Override
    public int getOutputCount() {
        // mock as output is only 1 dimension
        return 1;
    }

    public String getAlgorithm() {
        return this.getIndependentTreeModel().getAlgorithm();
    }

    public String getLossStr() {
        return this.getIndependentTreeModel().getLossStr();
    }

    public List<TreeNode> getTrees() {
        return this.getIndependentTreeModel().getTrees().get(0);
    }

    public boolean isGBDT() {
        return this.getIndependentTreeModel().isGBDT();
    }

    public boolean isClassfication() {
        return this.getIndependentTreeModel().isClassification();
    }

    public IndependentTreeModel getIndependentTreeModel() {
        return independentTreeModel;
    }

    /**
     * Get feature importance of current model.
     * 
     * @return map of feature importance, key is column index.
     */
    public Map<Integer, MutablePair<String, Double>> getFeatureImportances() {
        Map<Integer, MutablePair<String, Double>> importancesSum = new HashMap<Integer, MutablePair<String, Double>>();
        Map<Integer, String> nameMapping = this.getIndependentTreeModel().getNumNameMapping();
        int treeSize = this.getIndependentTreeModel().getTrees().size();

        // such case we only support treeModel is one element list
        if(this.getIndependentTreeModel().getTrees().size() != 1) {
            throw new RuntimeException(
                    "Bagging model cannot be supported in Tree Model one element feature importance computing.");
        }

        for(TreeNode tree: this.getIndependentTreeModel().getTrees().get(0)) {
            // get current tree importance at first
            Map<Integer, Double> subImportances = tree.computeFeatureImportance();
            // merge feature importance from different trees
            for(Entry<Integer, Double> entry: subImportances.entrySet()) {
                String featureName = nameMapping.get(entry.getKey());
                MutablePair<String, Double> importance = MutablePair.of(featureName, entry.getValue());
                if(!importancesSum.containsKey(entry.getKey())) {
                    importance.setValue(importance.getValue() / treeSize);
                    importancesSum.put(entry.getKey(), importance);
                } else {
                    MutablePair<String, Double> current = importancesSum.get(entry.getKey());
                    current.setValue(current.getValue() + importance.getValue() / treeSize);
                    importancesSum.put(entry.getKey(), current);
                }
            }
        }
        return importancesSum;
    }

    /**
     * Sort by feature importance.
     * 
     * @param unsortMap
     *            map of raw feature importance
     * @param order
     *            descending or ascending
     * @return map of feature importance, key is column index.
     */
    public static Map<Integer, MutablePair<String, Double>> sortByValue(
            Map<Integer, MutablePair<String, Double>> unsortMap, final boolean order) {
        List<Entry<Integer, MutablePair<String, Double>>> list = new LinkedList<Entry<Integer, MutablePair<String, Double>>>(
                unsortMap.entrySet());
        Collections.sort(list, new Comparator<Entry<Integer, MutablePair<String, Double>>>() {
            public int compare(Entry<Integer, MutablePair<String, Double>> o1,
                    Entry<Integer, MutablePair<String, Double>> o2) {
                if(order) {
                    return o1.getValue().getValue().compareTo(o2.getValue().getValue());
                } else {
                    return o2.getValue().getValue().compareTo(o1.getValue().getValue());
                }
            }
        });
        // Maintaining insertion order with the help of LinkedList
        Map<Integer, MutablePair<String, Double>> sortedMap = new LinkedHashMap<Integer, MutablePair<String, Double>>();
        for(Entry<Integer, MutablePair<String, Double>> entry: list) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }
        return sortedMap;
    }

    @Override
    public String toString() {
        return this.getIndependentTreeModel().getTrees().toString();
    }
}