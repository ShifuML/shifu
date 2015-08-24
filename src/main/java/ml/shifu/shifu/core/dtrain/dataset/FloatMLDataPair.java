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
package ml.shifu.shifu.core.dtrain.dataset;

import org.encog.ml.data.MLDataPair;

/**
 * Copy from {@link MLDataPair} to support float type data.
 */
public interface FloatMLDataPair {

    /**
     * @return The ideal data that the machine learning method should produce
     *         for the specified input.
     */
    float[] getIdealArray();

    /**
     * @return The input that the neural network
     */
    float[] getInputArray();

    /**
     * Set the ideal data, the desired output.
     * 
     * @param data
     *            The ideal data.
     */
    void setIdealArray(float[] data);

    /**
     * Set the input.
     * 
     * @param data
     *            The input.
     */
    void setInputArray(float[] data);

    /**
     * @return True if this training pair is supervised. That is, it has both
     *         input and ideal data.
     */
    boolean isSupervised();

    /**
     * @return The ideal data that the neural network should produce for the
     *         specified input.
     */
    FloatMLData getIdeal();

    /**
     * @return The input that the neural network
     */
    FloatMLData getInput();

    /**
     * Get the significance, 1.0 is neutral.
     * 
     * @return The significance.
     */
    float getSignificance();

    /**
     * Set the significance, 1.0 is neutral.
     * 
     * @param s
     *            The significance.
     */
    void setSignificance(float s);

}
