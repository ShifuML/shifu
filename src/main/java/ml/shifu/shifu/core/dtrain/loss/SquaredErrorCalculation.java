/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.loss;

/**
 * Squared error computation logic.
 */
public class SquaredErrorCalculation implements ml.shifu.shifu.core.dtrain.loss.ErrorCalculation {

    /**
     * The overall error.
     */
    private double globalError;

    /**
     * The size of a set.
     */
    private int setSize;

    /**
     * Returns the root mean square error for a complete training set.
     * 
     * @return The current error for the neural network.
     */
    @Override
    public final double calculate() {
        return this.globalError;
    }

    /**
     * Reset the error accumulation to zero.
     */
    @Override
    public final void reset() {
        this.globalError = 0;
        this.setSize = 0;
    }

    /**
     * Update the error with single values.
     * 
     * @param actual
     *            The actual value.
     * @param ideal
     *            The ideal value.
     */
    @Override
    public final double updateError(final double actual, final double ideal) {
        double delta = ideal - actual;
        double currentError = delta * delta;
        this.globalError += currentError;
        this.setSize += 1;
        return currentError;
    }

    /**
     * Called to update for each number that should be checked.
     * 
     * @param actual
     *            The actual number.
     * @param ideal
     *            The ideal number.
     */
    @Override
    public final void updateError(final double[] actual, final double[] ideal, final double significance) {
        for(int i = 0; i < actual.length; i++) {
            double delta = (ideal[i] - actual[i]);
            this.globalError += delta * delta * significance;
        }
        this.setSize += ideal.length;
    }

    @Override
    public int getSetSize() {
        return setSize;
    }

}
