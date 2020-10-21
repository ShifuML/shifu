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
package ml.shifu.shifu.core.dtrain.dataset;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.NeuralNetworkError;

/**
 * {@link CacheBasicFloatNetwork} is to cache first layer of sum and then in sensitivity analysis to use sum minues
 * current remove item. Details, please see {@link CacheFlatNetwork}.
 * 
 * Thanks (Chen Yang)ychen26@paypal.com to share such optimization idea.
 * 
 * @author pengzhang
 */
public class CacheBasicFloatNetwork extends BasicFloatNetwork {

    private static final long serialVersionUID = -8246909954154956217L;

    private BasicFloatNetwork network;

    public CacheBasicFloatNetwork(BasicFloatNetwork network) {
        this.network = network;
    }

    /**
     * Compute network score (forward computing). If cacheInputOutput is true, to cache first layer output in this
     * class. Then if cacheInputOutput is false, read value from cache and then use sum-current item to save CPU
     * computation.
     * 
     * @param input
     *            input value array
     * @param output
     *            output value
     * @param cacheInputOutput
     *            if it is to cache first layer output or to use first layer output cache.
     * @param resetInputIndex
     *            if cacheInputOutput is false, resetInputIndex is which item should be removed.
     */
    public void compute(double[] input, double[] output, boolean cacheInputOutput, int resetInputIndex) {
        CacheFlatNetwork flat = (CacheFlatNetwork) getFlat();
        flat.compute(input, output, cacheInputOutput, resetInputIndex);
    }

    /**
     * Compute network score (forward computing). If cacheInputOutput is true, to cache first layer output in this
     * class. Then if cacheInputOutput is false, read value from cache and then use sum-current item to save CPU
     * computation.
     * 
     * @param input
     *            input value array
     * @param cacheInputOutput
     *            if it is to cache first layer output or to use first layer output cache.
     * @param resetInputIndex
     *            if cacheInputOutput is false, resetInputIndex is which item should be removed.
     * @return output value as score.
     */
    public final MLData compute(final MLData input, boolean cacheInputOutput, int resetInputIndex) {
        try {
            final MLData result = new BasicMLData(this.network.getStructure().getFlat().getOutputCount());
            compute(input.getData(), result.getData(), cacheInputOutput, resetInputIndex);
            return result;
        } catch (final ArrayIndexOutOfBoundsException ex) {
            throw new NeuralNetworkError(
                    "Index exception: there was likely a mismatch between layer sizes, or the size of the input presented to the network.",
                    ex);
        }
    }

}
