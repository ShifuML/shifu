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
package ml.shifu.shifu.core.dtrain.wnd;

/**
 * This is single {@link Layer} abstraction for all kind of operations in a neural network graph. Typical implementation
 * like dense layer and activation, some others like embedding layer and wide field layer use the same interface for
 * graph level computation.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public interface Layer<FIN, FOUT, BIN, BOUT> {

    /**
     * Output dimesion of current layer.
     */
    public int getOutDim();

    /**
     * Layer forward computation from input to output. Activation in neural network is abstracted as one layer.
     * 
     * @param input
     *            the input object (float array or sparse input) of current layer
     * @return output results, common case like float[] or float.
     */
    public FOUT forward(FIN input);

    /**
     * Backward is used to backward compute inputs according to layer outputs. At the same time layer inside gradients
     * are computed together for model training/updates.
     * 
     * @param backInput
     *            the backward input from last layer backward.
     * @param significance
     *            weight/significance per each record in backward computation.
     * @return backward output result which is typically corresponding like input of layer.
     */
    public BOUT backward(BIN backInput, float significance);

}
