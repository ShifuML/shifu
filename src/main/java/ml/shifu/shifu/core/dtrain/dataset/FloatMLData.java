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

import org.encog.ml.data.MLData;

/**
 * Copy from {@link MLData} to support float type data.
 */
public interface FloatMLData extends Cloneable {

    /**
     * Add a value to the specified index.
     * 
     * @param index
     *            The index to add to.
     * @param value
     *            The value to add.
     */
    void add(int index, float value);

    /**
     * Clear any data to zero.
     */
    void clear();

    /**
     * Clone this object.
     * 
     * @return A cloned version of this object.
     */
    FloatMLData clone();

    /**
     * @return All of the elements as an array.
     */
    float[] getData();

    /**
     * Get the element specified index value.
     * 
     * @param index
     *            The index to read.
     * @return The value at the specified index.
     */
    float getData(int index);

    /**
     * Set all of the data as an array of floats.
     * 
     * @param data
     *            An array of floats.
     */
    void setData(float[] data);

    /**
     * Set the specified element.
     * 
     * @param index
     *            The index to set.
     * @param d
     *            The data for the specified element.
     */
    void setData(int index, float d);

    /**
     * @return How many elements are stored in this object.
     */
    int size();

}
