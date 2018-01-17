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
package ml.shifu.shifu.core.dtrain.nn;

/**
 * Copy from Encog Error Calculation to support log error, absolute error and squared error.
 */
public interface ErrorCalculation {

    /**
     * Return set size used to compute error.
     * 
     * @return the data set size
     */
    public int getSetSize();

    /**
     * Returns the root mean square error for a complete training set.
     * 
     * @return The current error for the neural network.
     */
    public double calculate();

    /**
     * Reset the error accumulation to zero.
     */
    public void reset();

    /**
     * Update the error with single values.
     * 
     * @param actual
     *            The actual value.
     * @param ideal
     *            The ideal value.
     */
    public void updateError(final double actual, final double ideal);

    /**
     * Called to update for each number that should be checked.
     * 
     * @param actual
     *            The actual number.
     * @param ideal
     *            The ideal number.
     * @param significance
     *            weight of the record
     */
    public void updateError(final double[] actual, final double[] ideal, final double significance);

}
