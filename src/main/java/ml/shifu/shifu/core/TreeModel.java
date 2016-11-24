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
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;

import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

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

    public static TreeModel loadFromStream(InputStream input, List<ColumnConfig> columnConfigList) throws IOException {
        return loadFromStream(input, columnConfigList, false);
    }

    public static TreeModel loadFromStream(InputStream input, List<ColumnConfig> columnConfigList,
            boolean isConvertToProb) throws IOException {
        return new TreeModel(IndependentTreeModel.loadFromStream(input, isConvertToProb));
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

}