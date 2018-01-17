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
package ml.shifu.shifu.core.pmml.builder.impl;

import ml.shifu.shifu.core.pmml.PMMLEncogNeuralNetworkModel;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import org.dmg.pmml.*;
import org.encog.ml.BasicML;
import org.encog.neural.networks.BasicNetwork;

/**
 * Created by zhanhu on 3/29/16.
 */
public class NNSpecifCreator extends AbstractSpecifCreator {

    @Override
    public boolean build(BasicML basicML, Model model) {
        NeuralNetwork nnPmmlModel = (NeuralNetwork) model;
        new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML((BasicNetwork) basicML, nnPmmlModel);
        nnPmmlModel.withOutput(createNormalizedOutput());
        return true;
    }

    @Override
    public boolean build(BasicML basicML, Model model, int id) {
        NeuralNetwork nnPmmlModel = (NeuralNetwork) model;
        new PMMLEncogNeuralNetworkModel().adaptMLModelToPMML((BasicNetwork) basicML, nnPmmlModel);
        nnPmmlModel.withOutput(createNormalizedOutput(id));
        return true;
    }
}
