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

import org.apache.commons.lang3.tuple.MutablePair;
import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;

/**
 * {@link TreeModel} is to load Random Forest or Gradient Boosted Decision Tree models.
 * 
 * <p>
 * {@link #loadFromStream(InputStream, ModelConfig, List)} can be used to read serialized models.
 * 
 * <p>
 * TODO, make trees computing in parallel
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class TreeModel extends BasicML implements MLRegression {

    private static final long serialVersionUID = 479043597958785224L;

    private IndependentTreeModel independentTreeModel;

    public TreeModel(IndependentTreeModel independentTreeModel) {
        this.independentTreeModel = independentTreeModel;
    }

    @Override
    public final MLData compute(final MLData input) {
        double[] data = input.getData();
        return new BasicMLData(this.getIndependentTreeModel().compute(data));
    }

    @Override
    public int getInputCount() {
        return this.getIndependentTreeModel().getInputNode();
    }

    @Override
    public String toString() {
        return this.getIndependentTreeModel().getTrees().toString();
    }

    @Override
    public void updateProperties() {
        // No need implementation
    }

    public static TreeModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, false);
    }

    public static TreeModel loadFromStream(InputStream input, boolean isConvertToProb) throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb));
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.encog.ml.MLOutput#getOutputCount()
     */
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
        return this.getIndependentTreeModel().getTrees();
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

    public Map<Integer, MutablePair<String, Double>> getFeatureImportances() {
        Map<Integer, MutablePair<String, Double>> importancesSum = new HashMap<Integer, MutablePair<String, Double>>();
        Map<Integer, String> nameMapping = this.getIndependentTreeModel().getNumNameMapping();
        int size = this.getIndependentTreeModel().getTrees().size();
        for(TreeNode tree: this.getIndependentTreeModel().getTrees()) {
            Map<Integer, Double> subImportances = tree.computeFeatureImportance();
            for(Entry<Integer, Double> entry: subImportances.entrySet()) {
                String featureName = nameMapping.get(entry.getKey());
                MutablePair<String, Double> importance = MutablePair.of(featureName, entry.getValue());
                if(!importancesSum.containsKey(entry.getKey())) {
                    importance.setValue(importance.getValue() / size);
                    importancesSum.put(entry.getKey(), importance);
                } else {
                    MutablePair<String, Double> current = importancesSum.get(entry.getKey());
                    current.setValue(current.getValue() + importance.getValue() / size);
                    importancesSum.put(entry.getKey(), current);
                }
            }
        }
        return importancesSum;
    }

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
}