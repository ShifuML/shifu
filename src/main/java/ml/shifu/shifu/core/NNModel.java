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

import ml.shifu.shifu.core.dtrain.nn.IndependentNNModel;

import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

/**
 * NN model wrappered to Encog interface for evaluation.
 * 
 * <p>
 * {@link #loadFromStream(InputStream, boolean)} can be used to read serialized models. Which is delegated to
 * {@link IndependentNNModel}.
 */
public class NNModel extends BasicML implements MLRegression {

    private static final long serialVersionUID = 479043597958785224L;

    /**
     * Independent model instance
     */
    private transient IndependentNNModel independentNNModel;

    /**
     * Constructor on current {@link IndependentNNModel}
     * 
     * @param independentNNModel
     *            the independent nn model
     */
    public NNModel(IndependentNNModel independentNNModel) {
        this.setIndependentNNModel(independentNNModel);
    }

    /**
     * Compute model score based on given input double array.
     */
    @Override
    public final MLData compute(final MLData input) {
        double[] data = input.getData();
        return new BasicMLData(this.getIndependentNNModel().compute(data));
    }

    /**
     * How many input columns.
     */
    @Override
    public int getInputCount() {
        return this.getIndependentNNModel().getBasicNetworks().get(0).getInputCount();
    }

    @Override
    public void updateProperties() {
        // No need implementation
    }

    public static NNModel loadFromStream(InputStream input) throws IOException {
        return loadFromStream(input, true);
    }

    public static NNModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        return new NNModel(IndependentNNModel.loadFromStream(input, isRemoveNameSpace));
    }

    @Override
    public int getOutputCount() {
        // mock as output is only 1 dimension
        return this.getIndependentNNModel().getBasicNetworks().get(0).getOutputCount();
    }

    /**
     * @return the independentNNModel
     */
    public IndependentNNModel getIndependentNNModel() {
        return independentNNModel;
    }

    /**
     * @param independentNNModel
     *            the independentNNModel to set
     */
    public void setIndependentNNModel(IndependentNNModel independentNNModel) {
        this.independentNNModel = independentNNModel;
    }

}