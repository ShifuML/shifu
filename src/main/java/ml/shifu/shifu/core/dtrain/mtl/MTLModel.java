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

import java.io.IOException;
import java.io.InputStream;

import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

/**
 * {@link MTLModel} as multi-task model could be used in 'eval' step of Shifu with the same interface introduced by
 * encog {@link MLRegression}.
 * 
 * <p>
 * A basic multi-task model inference engine is created to support input double array and multi-task outputs.
 * 
 * <p>
 * {@link MTLModel} is delegated to {@link IndependentMTLModel} for model inference.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MTLModel extends BasicML implements MLRegression {

    /**
     * Serial UID
     */
    private static final long serialVersionUID = 1070575695157933853L;

    /**
     * 'independent' multi-task model instance without dependency on encog.
     */
    IndependentMTLModel independentMTLModel;

    /**
     * Constructor from proxy engine instance.
     * 
     * @param independentMTLModel
     *            the independent multi-task model instance
     */
    public MTLModel(IndependentMTLModel independentMTLModel) {
        this.independentMTLModel = independentMTLModel;
    }

    @Override
    public int getInputCount() {
        return independentMTLModel.getMtm().getDil().getOutDim();
    }

    @Override
    public int getOutputCount() {
        return independentMTLModel.getMtm().getFinalLayers().size();
    }

    /**
     * Basic inference entrance to compute multiple task outputs given inputs.
     * 
     * @param input
     *            the input double array wrapper by encog interface
     * @return multiple task outputs
     */
    @Override
    public MLData compute(MLData input) {
        double[] result = independentMTLModel.compute(input.getData());
        return new BasicMLData(result);
    }

    @Override
    public void updateProperties() {
        // No need to implement
    }

    /**
     * Load model instance from model spec stream.
     * 
     * @param input
     *            the model spec input stream, should be closed outside.
     * @return {@link MTLModel} instance
     * @throws IOException
     *             if any IO exception in loading input stream.
     */
    public static MTLModel loadFromStream(InputStream input) throws IOException {
        return new MTLModel(IndependentMTLModel.loadFromStream(input));
    }

    /**
     * Load model instance from model spec stream.
     * 
     * @param input
     *            the model spec input stream, should be closed outside.
     * @param isRemoveNameSpace
     *            if need remove name space of column or not
     * @return {@link MTLModel} instance
     * @throws IOException
     *             if any IO exception in loading input stream.
     */
    public static MTLModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        return new MTLModel(IndependentMTLModel.loadFromStream(input, isRemoveNameSpace));
    }

}
