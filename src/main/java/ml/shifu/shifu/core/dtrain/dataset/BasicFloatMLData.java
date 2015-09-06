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

import org.encog.ml.data.basic.BasicMLData;

/**
 * Copy from {@link BasicMLData} to support float type data.
 */
public class BasicFloatMLData implements FloatMLData, Serializable, Cloneable {

    /**
     * The serial id.
     */
    private static final long serialVersionUID = -3644304891793584603L;

    /**
     * The data held by this object.
     */
    private float[] data;

    /**
     * Construct this object with the specified data.
     * 
     * @param d
     *            The data to construct this object with.
     */
    public BasicFloatMLData(final float[] d) {
        this(d.length);
        System.arraycopy(d, 0, this.data, 0, d.length);
    }

    /**
     * Construct this object with blank data and a specified size.
     * 
     * @param size
     *            The amount of data to store.
     */
    public BasicFloatMLData(final int size) {
        this.data = new float[size];
    }

    /**
     * Construct a new BasicFloatMLData object from an existing one. This makes a
     * copy of an array.
     * 
     * @param d
     *            The object to be copied.
     */
    public BasicFloatMLData(final FloatMLData d) {
        this(d.size());
        System.arraycopy(d.getData(), 0, this.data, 0, d.size());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void add(final int index, final float value) {
        this.data[index] += value;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void clear() {
        for(int i = 0; i < this.data.length; i++) {
            this.data[i] = 0;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final FloatMLData clone() {
        return new BasicFloatMLData(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final float[] getData() {
        return this.data;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final float getData(final int index) {
        return this.data[index];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setData(final float[] theData) {
        this.data = theData;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setData(final int index, final float d) {
        this.data[index] = d;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final int size() {
        return this.data.length;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final String toString() {
        final StringBuilder builder = new StringBuilder("[");
        builder.append(this.getClass().getSimpleName());
        builder.append(":");
        for(int i = 0; i < this.data.length; i++) {
            if(i != 0) {
                builder.append(',');
            }
            builder.append(this.data[i]);
        }
        builder.append("]");
        return builder.toString();
    }

}