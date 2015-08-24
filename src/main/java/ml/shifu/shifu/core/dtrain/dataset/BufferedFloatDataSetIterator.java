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

import java.util.Iterator;

import org.encog.ml.data.buffer.BufferedDataSetIterator;

/**
 * Copy from {@link BufferedDataSetIterator} to support float type data.
 */
public class BufferedFloatDataSetIterator implements Iterator<FloatMLDataPair> {

    /**
     * The dataset being iterated over.
     */
    private final BufferedFloatMLDataSet data;

    /**
     * The current record.
     */
    private int current;

    /**
     * Construct the iterator.
     * 
     * @param theData
     *            The dataset to iterate over.
     */
    public BufferedFloatDataSetIterator(final BufferedFloatMLDataSet theData) {
        this.data = theData;
        this.current = 0;
    }

    /**
     * @return True if there is are more records to read.
     */
    @Override
    public final boolean hasNext() {
        return this.current < this.data.getRecordCount();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final FloatMLDataPair next() {
        if (!hasNext()) {
            return null;
        }

        final FloatMLDataPair pair = BasicFloatMLDataPair.createPair(
                this.data.getInputSize(), this.data.getIdealSize());
        this.data.getRecord(this.current++, pair);
        return pair;
    }

    /**
     * Not supported.
     */
    @Override
    public final void remove() {
        throw new RuntimeException("Remove is not supported.");
    }

}