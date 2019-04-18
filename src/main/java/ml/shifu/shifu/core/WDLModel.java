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
package ml.shifu.shifu.core;

import java.io.IOException;
import java.io.InputStream;

import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLData;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLData;
import ml.shifu.shifu.core.dtrain.wdl.FloatMLRegression;
import ml.shifu.shifu.core.dtrain.wdl.IndependentWDLModel;

/**
 * {@link WDLModel} is to load WideAndDeep models from Encog interfaces. If user wouldn't like to depend on encog,
 * {@link IndependentWDLModel} can be used to execute WideAndDeep model from Shifu.
 * 
 * @author juguo
 */
public class WDLModel extends BasicML implements FloatMLRegression {

    /**
     * WDL model instance without dependency on encog.
     */
    IndependentWDLModel independentWDLModel;

    public WDLModel(IndependentWDLModel independentWDLModel) {
        this.independentWDLModel = independentWDLModel;
    }

    @Override
    public int getInputCount() {
        return independentWDLModel.getWnd().getInputNum();
    }

    @Override
    public int getOutputCount() {
        return 1;
    }

    @Override
    public MLData compute(MLData input) {
        double[] data = input.getData();
        float[] result = independentWDLModel.compute(data);
        double[] dRes = new double[result.length];
        for(int i = 0; i < result.length; i++) {
            dRes[i] = result[i];
        }
        return new BasicMLData(dRes);
    }

    @Override
    public FloatMLData compute(FloatMLData input) {
        return new BasicFloatMLData(independentWDLModel.compute(input.getData()));
    }

    @Override
    public void updateProperties() {
        // No need to implement

    }

    public static WDLModel loadFromStream(InputStream input) throws IOException {
        return new WDLModel(IndependentWDLModel.loadFromStream(input));
    }

    public static WDLModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        return new WDLModel(IndependentWDLModel.loadFromStream(input, isRemoveNameSpace));
    }

}
