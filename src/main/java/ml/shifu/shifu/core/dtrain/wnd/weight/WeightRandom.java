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
package ml.shifu.shifu.core.dtrain.wnd.weight;

import ml.shifu.shifu.core.dtrain.random.AbstractWeightRandomizer;

/**
 * Class Description.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class WeightRandom implements Initialisable {
    private AbstractWeightRandomizer randomizer;
    private static final float NO_USE = 666f;

    public WeightRandom(AbstractWeightRandomizer randomizer) {
        this.randomizer = randomizer;
    }

    /**
     * Init with one float number.
     *
     * @return the init number.
     */
    @Override
    public float initWeight() {
        return this.randomizer.randomize(NO_USE);
    }

    /**
     * Init a one dimensional float array with specific length
     *
     * @param length@return the init array
     */
    @Override
    public float[] initWeight(int length) {
        float[] weight = new float[length];
        this.randomizer.randomize(weight);
        return weight;
    }

    /**
     * Init a two dimensional float array with specific row and column
     *
     * @param row the row number
     * @param col the column number
     * @return the init array
     */
    @Override
    public float[][] initWeight(int row, int col) {
        float[][] weight = new float[row][col];
        this.randomizer.randomize(weight);
        return weight;
    }

}
