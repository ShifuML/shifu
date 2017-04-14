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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.encog.util.obj.ObjectCloner;

/**
 * Copy from {@link BasicFloatMLDataSet} to support float type data.
 */
public class BasicFloatMLDataSet implements Serializable, FloatMLDataSet, Cloneable {

    /**
     * An iterator to be used with the BasicFloatMLDataSet. This iterator does not
     * support removes.
     */
    public class BasicMLIterator implements Iterator<FloatMLDataPair> {

        /**
         * The index that the iterator is currently at.
         */
        private int currentIndex = 0;

        /**
         * {@inheritDoc}
         */
        @Override
        public final boolean hasNext() {
            return this.currentIndex < BasicFloatMLDataSet.this.data.size();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public final FloatMLDataPair next() {
            if(!hasNext()) {
                return null;
            }

            return BasicFloatMLDataSet.this.data.get(this.currentIndex++);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public final void remove() {
            throw new RuntimeException("Called remove, unsupported operation.");
        }
    }

    /**
     * The serial id.
     */
    private static final long serialVersionUID = -2279722928570071183L;

    /**
     * The data held by this object.
     */
    private List<FloatMLDataPair> data = new ArrayList<FloatMLDataPair>();

    /**
     * Default constructor.
     */
    public BasicFloatMLDataSet() {
    }

    /**
     * Construct a data set from an input and idea array.
     * 
     * @param input
     *            The input into the machine learning method for training.
     * @param ideal
     *            The ideal output for training.
     */
    public BasicFloatMLDataSet(final float[][] input, final float[][] ideal) {
        if(ideal != null) {
            for(int i = 0; i < input.length; i++) {
                final BasicFloatMLData inputData = new BasicFloatMLData(input[i]);
                final BasicFloatMLData idealData = new BasicFloatMLData(ideal[i]);
                this.add(inputData, idealData);
            }
        } else {
            for(final float[] element: input) {
                final BasicFloatMLData inputData = new BasicFloatMLData(element);
                this.add(inputData);
            }
        }
    }

    /**
     * Construct a data set from an already created list. Mostly used to
     * duplicate this class.
     * 
     * @param theData
     *            The data to use.
     */
    public BasicFloatMLDataSet(final List<FloatMLDataPair> theData) {
        this.data = theData;
    }

    /**
     * Copy whatever dataset type is specified into a memory dataset.
     * 
     * @param set
     *            The dataset to copy.
     */
    public BasicFloatMLDataSet(final FloatMLDataSet set) {
        final int inputCount = set.getInputSize();
        final int idealCount = set.getIdealSize();

        for(final FloatMLDataPair pair: set) {

            BasicFloatMLData input = null;
            BasicFloatMLData ideal = null;

            if(inputCount > 0) {
                input = new BasicFloatMLData(inputCount);
                BasicFloatMLDataSet.arrayCopy(pair.getInputArray(), input.getData());
            }

            if(idealCount > 0) {
                ideal = new BasicFloatMLData(idealCount);
                BasicFloatMLDataSet.arrayCopy(pair.getIdealArray(), ideal.getData());
            }

            add(new BasicFloatMLDataPair(input, ideal));
        }
    }

    /**
     * Copy an array of floats to an array of floats.
     * 
     * @param source
     *            The source array.
     * @param target
     *            The target array.
     */
    public static void arrayCopy(final float[] source, final float[] target) {
        for(int i = 0; i < source.length; i++) {
            target[i] = source[i];
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(final FloatMLData theData) {
        this.data.add(new BasicFloatMLDataPair(theData));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(final FloatMLData inputData, final FloatMLData idealData) {
        final FloatMLDataPair pair = new BasicFloatMLDataPair(inputData, idealData);
        this.data.add(pair);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(final FloatMLDataPair inputData) {
        this.data.add(inputData);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final Object clone() {
        return ObjectCloner.deepCopy(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void close() {
        // nothing to close
    }

    /**
     * Get the data held by this container.
     * 
     * @return the data
     */
    public final List<FloatMLDataPair> getData() {
        return this.data;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final int getIdealSize() {
        if(this.data.isEmpty()) {
            return 0;
        }
        final FloatMLDataPair first = this.data.get(0);
        if(first.getIdeal() == null) {
            return 0;
        }

        return first.getIdeal().size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final int getInputSize() {
        if(this.data.isEmpty()) {
            return 0;
        }
        final FloatMLDataPair first = this.data.get(0);
        return first.getInput().size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void getRecord(final long index, final FloatMLDataPair pair) {
        final FloatMLDataPair source = this.data.get((int) index);
        pair.setInputArray(source.getInputArray());
        if(pair.getIdealArray() != null) {
            pair.setIdealArray(source.getIdealArray());
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final long getRecordCount() {
        return this.data.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final boolean isSupervised() {
        if(this.data.size() == 0) {
            return false;
        }
        return this.data.get(0).isSupervised();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final Iterator<FloatMLDataPair> iterator() {
        final BasicMLIterator result = new BasicMLIterator();
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final FloatMLDataSet openAdditional() {
        return new BasicFloatMLDataSet(this.data);
    }

    /**
     * @param theData
     *            the data to set
     */
    public final void setData(final List<FloatMLDataPair> theData) {
        this.data = theData;
    }

}
