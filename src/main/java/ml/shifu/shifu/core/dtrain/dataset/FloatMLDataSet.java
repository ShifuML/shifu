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

import org.encog.ml.data.MLDataSet;

/**
 * Copy from {@link MLDataSet} to support float type data.
 */
public interface FloatMLDataSet extends Iterable<FloatMLDataPair> {

    /**
     * @return The size of the input data.
     */
    int getIdealSize();

    /**
     * @return The size of the input data.
     */
    int getInputSize();

    /**
     * @return True if this is a supervised training set.
     */
    boolean isSupervised();

    /**
     * Determine the total number of records in the set.
     * 
     * @return The total number of records in the set.
     */
    long getRecordCount();

    /**
     * Read an individual record, specified by index, in random order.
     * 
     * @param index
     *            The index to read.
     * @param pair
     *            The pair that the record will be copied into.
     */
    void getRecord(long index, FloatMLDataPair pair);

    /**
     * Opens an additional instance of this dataset.
     * 
     * @return The new instance.
     */
    FloatMLDataSet openAdditional();

    /**
     * Add a object to the dataset. This is used with unsupervised training, as
     * no ideal output is provided. Note: not all implemenations support the add
     * methods.
     * 
     * @param data
     *            The data item to be added.
     */
    void add(FloatMLData data);

    /**
     * Add a set of input and ideal data to the dataset. This is used with
     * supervised training, as ideal output is provided. Note: not all
     * implementations support the add methods.
     * 
     * @param inputData
     *            Input data.
     * @param idealData
     *            Ideal data.
     */
    void add(FloatMLData inputData, FloatMLData idealData);

    /**
     * Add a an object to the dataset. This is used with unsupervised training,
     * as no ideal output is provided. Note: not all implementations support the
     * add methods.
     * 
     * @param inputData
     *            A MLDataPair object that contains both input and ideal data.
     */
    void add(FloatMLDataPair inputData);

    /**
     * Close this datasource and release any resources obtained by it, including
     * any iterators created.
     */
    void close();
}