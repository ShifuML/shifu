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
package ml.shifu.shifu.core.dtrain.wdl.activation;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.shifu.core.dtrain.wdl.Layer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * {@link Activation} is basic abstraction for different kind of activation methods like sigmoid, ReLU, tanh ...
 *
 * Empty construction methods is required for it's implementation class. This will be used to create reflection
 * initialize this type of activation.
 *
 * @author Zhang David (pengzhang@paypal.com)
 */
public abstract class Activation implements Layer<double[], double[], double[], double[]>, Bytable {

    @Override
    public int getOutDim() {
        // no need call output dimension, as Activation usually for 1:1 mapping between inputs and outputs.
        throw new UnsupportedOperationException();
    }

    @Override
    public void write(DataOutput out) throws IOException {
    }

    @Override
    public void readFields(DataInput in) throws IOException {
    }

}
