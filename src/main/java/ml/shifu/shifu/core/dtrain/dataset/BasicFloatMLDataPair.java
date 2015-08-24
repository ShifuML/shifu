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

import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.util.Format;

/**
 * Copy from {@link BasicMLDataPair} to support float type data.
 */
public class BasicFloatMLDataPair implements FloatMLDataPair, Serializable {

    /**
     * The serial ID.
     */
    private static final long serialVersionUID = -9068229682273861359L;

    /**
     * The significance.
     */
    private float significance = 1.0f;

    /**
     * Create a new data pair object of the correct size for the machine
     * learning method that is being trained. This object will be passed to the
     * getPair method to allow the data pair objects to be copied to it.
     * 
     * @param inputSize
     *            The size of the input data.
     * @param idealSize
     *            The size of the ideal data.
     * @return A new data pair object.
     */
    public static FloatMLDataPair createPair(final int inputSize, final int idealSize) {
        FloatMLDataPair result;

        if(idealSize > 0) {
            result = new BasicFloatMLDataPair(new BasicFloatMLData(inputSize), new BasicFloatMLData(idealSize));
        } else {
            result = new BasicFloatMLDataPair(new BasicFloatMLData(inputSize));
        }

        return result;
    }

    /**
     * The the expected output from the machine learning method, or null for
     * unsupervised training.
     */
    private final FloatMLData ideal;

    /**
     * The training input to the machine learning method.
     */
    private final FloatMLData input;

    /**
     * Construct the object with only input. If this constructor is used, then
     * unsupervised training is being used.
     * 
     * @param theInput
     *            The input to the machine learning method.
     */
    public BasicFloatMLDataPair(final FloatMLData theInput) {
        this.input = theInput;
        this.ideal = null;
    }

    /**
     * Construct a BasicFloatMLDataPair class with the specified input and ideal
     * values.
     * 
     * @param theInput
     *            The input to the machine learning method.
     * @param theIdeal
     *            The expected results from the machine learning method.
     */
    public BasicFloatMLDataPair(final FloatMLData theInput, final FloatMLData theIdeal) {
        this.input = theInput;
        this.ideal = theIdeal;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final FloatMLData getIdeal() {
        return this.ideal;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final float[] getIdealArray() {
        if(this.ideal == null) {
            return null;
        }
        return this.ideal.getData();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final FloatMLData getInput() {
        return this.input;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final float[] getInputArray() {
        return this.input.getData();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final boolean isSupervised() {
        return this.ideal != null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setIdealArray(final float[] data) {
        this.ideal.setData(data);

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void setInputArray(final float[] data) {
        this.input.setData(data);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final String toString() {
        final StringBuilder builder = new StringBuilder("[");
        builder.append(this.getClass().getSimpleName());
        builder.append(":");
        builder.append("Input:");
        builder.append(getInput());
        builder.append("Ideal:");
        builder.append(getIdeal());
        builder.append(",");
        builder.append("Significance:");
        builder.append(Format.formatPercent(this.significance));
        builder.append("]");
        return builder.toString();
    }

    /**
     * {@inheritDoc}
     */
    public float getSignificance() {
        return significance;
    }

    /**
     * {@inheritDoc}
     */
    public void setSignificance(float significance) {
        this.significance = significance;
    }

}
