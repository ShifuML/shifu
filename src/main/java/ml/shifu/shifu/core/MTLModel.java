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

import ml.shifu.shifu.core.dtrain.mtl.IndependentMTLModel;
import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import java.io.IOException;
import java.io.InputStream;

/**
 * @author haillu
 */
public class MTLModel extends BasicML implements MLRegression {
    private IndependentMTLModel independentMTLModel;

    public MTLModel(IndependentMTLModel independentMTLModel) {
        this.independentMTLModel = independentMTLModel;
    }

    @Override
    public void updateProperties() {
        // No need to implement
    }

    @Override
    public MLData compute(MLData input) {
        double[] result = independentMTLModel.compute(input.getData());
        return new BasicMLData(result);
    }

    @Override
    public int getInputCount() {
        return independentMTLModel.getMtl().getInputSize();
    }

    @Override
    public int getOutputCount() {
        return independentMTLModel.getMtl().getTaskNumber();
    }

    public static MTLModel loadFromStream(InputStream input) throws IOException {
        return new MTLModel(IndependentMTLModel.loadFromStream(input));
    }

    public static MTLModel loadFromStream(InputStream input, boolean isRemoveNameSpace) throws IOException {
        return new MTLModel(IndependentMTLModel.loadFromStream(input, isRemoveNameSpace));
    }

}
