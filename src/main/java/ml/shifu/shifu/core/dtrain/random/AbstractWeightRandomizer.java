/*
 * Encog(tm) Core v3.4 - Java Version
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-core
 
 * Copyright 2008-2017 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package ml.shifu.shifu.core.dtrain.random;

import org.encog.mathutil.matrices.Matrix;
import org.encog.ml.MLMethod;
import org.encog.neural.networks.BasicNetwork;

/**
 * Copied from https://github.com/encog/encog-java-core/.
 */
public abstract class AbstractWeightRandomizer implements Randomizer {

    /**
     * The y2 value.
     */
    protected double y2;

    /**
     * Should we use the last value.
     */
    protected boolean useLast = false;

    protected GenerateRandom rnd;

    public AbstractWeightRandomizer() {
        this(System.currentTimeMillis());
    }

    public AbstractWeightRandomizer(long seed) {
        this.rnd = new MersenneTwisterGenerateRandom(seed);
    }

    /**
     * Generate a random number.
     * 
     * @param d
     *            The input value, not used.
     * @return The random number.
     */
    public double randomize(final double d) {
        return this.rnd.nextDouble();
    }

    /**
     * Randomize one level of a neural network.
     * 
     * @param network
     *            The network to randomize
     * @param fromLayer
     *            The from level to randomize.
     */
    public abstract void randomize(final BasicNetwork network, final int fromLayer);

    /**
     * Randomize the synapses and biases in the basic network based on an array,
     * modify the array. Previous values may be used, or they may be discarded,
     * depending on the randomizer.
     * 
     * @param method
     *            A network to randomize.
     */
    @Override
    public void randomize(final MLMethod method) {
        final BasicNetwork network = (BasicNetwork) method;
        for(int i = 0; i < network.getLayerCount() - 1; i++) {
            randomize(network, i);
        }
    }

    @Override
    public void randomize(double[] d) {
        randomize(d, 0, d.length);
    }

    @Override
    public void randomize(double[][] d) {
        for(int i = 0; i < d.length; i++) {
            for(int j = 0; j < d[j].length; j++) {
                d[i][j] = this.rnd.nextDouble();
            }
        }
    }

    @Override
    public void randomize(Matrix m) {
        randomize(m.getData());
    }

    @Override
    public void randomize(double[] d, int begin, int size) {
        for(int i = 0; i < size; i++) {
            d[begin + i] = this.rnd.nextDouble();
        }
    }

    @Override
    public float randomize(float f) {
        return this.rnd.nextFloat();
    }

    @Override
    public void randomize(float[] f) {
        randomize(f, 0, f.length);
    }

    @Override
    public void randomize(float[][] f) {
        for(int i = 0; i < f.length; i++) {
            for(int j = 0; j < f[j].length; j++) {
                f[i][j] = this.rnd.nextFloat();
            }
        }
    }

    @Override
    public void randomize(float[] f, int begin, int size) {
        for(int i = 0; i < size; i++) {
            f[begin + i] = this.rnd.nextFloat();
        }
    }

    public void setRandom(GenerateRandom theRandom) {
        this.rnd = theRandom;
    }

    public GenerateRandom getRandom() {
        return this.rnd;
    }

}