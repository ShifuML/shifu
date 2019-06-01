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
package ml.shifu.shifu.core.dtrain.wdl.weight;

import java.util.Random;

/**
 * A randomizer that will create random weight and bias values that are between a specified range.
 * <p>
 * Copied from https://github.com/encog/encog-java-core/.
 * </p>
 * @author jheaton
 */
public class RangeRandom implements Initialisable {

    /**
     * The random number generator.
     */
    private Random random;

    /**
     * The minimum value for the random range.
     */
    private final double min;

    /**
     * The maximum value for the random range.
     */
    private final double max;

    /**
     * Construct a range randomizer.
     *
     * @param min
     *            The minimum random value.
     * @param max
     *            The maximum random value.
     */
    public RangeRandom(final double min, final double max) {
        this.max = max;
        this.min = min;
        this.random = new Random(System.nanoTime());
    }

    /**
     * Generate a random number based on the range specified in the constructor.
     *
     * @return The random number.
     */
    public double randomize() {
        return nextdouble(this.min, this.max);
    }

    /**
     * Generate a random number in the specified range.
     *
     * @param min
     *            The minimum value.
     * @param max
     *            The maximum value.
     * @return A random number.
     */
    public final double nextdouble(final double min, final double max) {
        final double range = max - min;
        return (range * this.random.nextDouble()) + min;
    }

    @Override
    public double initWeight() {
        return randomize();
    }

    @Override
    public double[] initWeight(int length) {
        double[] weight = new double[length];
        randomize(weight);
        return weight;
    }

    @Override
    public double[][] initWeight(int row, int col) {
        double[][] weight = new double[row][col];
        randomize(weight);
        return weight;
    }

    /**
     * Randomize the array based on an array, modify the array. Previous values
     * may be used, or they may be discarded, depending on the randomizer.
     *
     * @param f
     *            An array to randomize.
     */
    public void randomize(final double[] f) {
        randomize(f, 0, f.length);
    }

    /**
     * Randomize the array based on an array, modify the array. Previous values
     * may be used, or they may be discarded, depending on the randomizer.
     *
     * @param f
     *            An array to randomize.
     * @param begin
     *            The beginning element of the array.
     * @param size
     *            The size of the array to copy.
     */
    public void randomize(final double[] f, final int begin,
                          final int size) {
        for (int i = 0; i < size; i++) {
            f[begin + i] = randomize();
        }
    }

    /**
     * Randomize the 2d array based on an array, modify the array. Previous
     * values may be used, or they may be discarded, depending on the
     * randomizer.
     *
     * @param f
     *            An array to randomize.
     */
    public void randomize(final double[][] f) {
        for (int r = 0; r < f.length; r++) {
            for (int c = 0; c < f[0].length; c++) {
                f[r][c] = randomize();
            }
        }
    }

    /**
     * @return the min
     */
    public double getMin() {
        return min;
    }

    /**
     * @return the max
     */
    public double getMax() {
        return max;
    }
}
