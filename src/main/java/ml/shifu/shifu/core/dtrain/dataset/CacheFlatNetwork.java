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

import org.encog.neural.flat.FlatLayer;
import org.encog.neural.flat.FlatNetwork;

/**
 * In sensitivity computing, the first layer sum values are computed and cached in this class, when reset each input
 * node,
 * use sum minus current item to save CPU computation.
 * 
 * <p>
 * First call should call compute method with cache true to cache input and then call compute with cache flag false to
 * leverage cache values in this item.
 * 
 * <p>
 * Thanks Chen Yang(ychen26@paypal.com) to share such optimization idea.
 */
public class CacheFlatNetwork extends FlatNetwork implements Cloneable {

    private static final long serialVersionUID = -7208969306860840672L;

    /**
     * Cache First layer of outputs. This array only is initialized in first call with cacheInputOutput set to true in
     * {@link #compute(double[], double[], boolean, int)}.
     */
    private double[] firstLayerCache;

    /**
     * Default constructor
     */
    public CacheFlatNetwork() {
    }

    /**
     * Constructor with layers.
     * 
     * @param layers
     *            the layers of such network.
     */
    public CacheFlatNetwork(final FlatLayer[] layers) {
        initNetwork(layers);
    }

    /**
     * Init internal variables for such network structure.
     * 
     * @param layers
     *            the layers of such network.
     */
    public void initNetwork(FlatLayer[] layers) {
        super.init(layers);
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
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            getLayerOutput()[i + sourceIndex] = input[i];
        }

        for(int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i, cacheInputOutput, resetInputIndex);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        System.arraycopy(getLayerOutput(), 0, output, 0, this.getOutputCount());
    }

    /**
     * Compute network score in each layer(forward computing). If cacheInputOutput is true, to cache first layer output
     * in this
     * class. Then if cacheInputOutput is false, read value from cache and then use sum-current item to save CPU
     * computation.
     * 
     * @param currentLayer
     *            current layer of network
     * @param cacheInputOutput
     *            if it is to cache first layer output or to use first layer output cache.
     * @param resetInputIndex
     *            if cacheInputOutput is false, resetInputIndex is which item should be removed.
     */
    protected void computeLayer(final int currentLayer, boolean cacheInputOutput, int resetInputIndex) {
        final int inputIndex = super.getLayerIndex()[currentLayer];
        final int outputIndex = super.getLayerIndex()[currentLayer - 1];
        final int inputSize = super.getLayerCounts()[currentLayer];
        final int outputSize = super.getLayerFeedCounts()[currentLayer - 1];

        int index = super.getWeightIndex()[currentLayer - 1];

        final int limitX = outputIndex + outputSize;
        final int limitY = inputIndex + inputSize;

        // init cache array for the first time
        if(cacheInputOutput && currentLayer == this.getLayerIndex().length - 1) {
            if(this.firstLayerCache == null || this.firstLayerCache.length != outputSize) {
                this.firstLayerCache = new double[outputSize];
            }
        }

        // weight values
        for(int x = outputIndex; x < limitX; x++) {
            if(cacheInputOutput) {
                double sum = 0;
                for(int y = inputIndex; y < limitY; y++) {
                    sum += super.getWeights()[index++] * super.getLayerOutput()[y];
                }

                super.getLayerSums()[x] = sum;
                super.getLayerOutput()[x] = sum;

                if(currentLayer == this.getLayerIndex().length - 1) {
                    // cache first layer output here
                    this.firstLayerCache[x - outputIndex] = sum;
                }
            } else {
                if(currentLayer == this.getLayerIndex().length - 1) {
                    // to use the cache, use cache_sum - current item value to save computation to save computation
                    double sum = this.firstLayerCache[x - outputIndex] - super.getWeights()[index + resetInputIndex]
                            * super.getLayerOutput()[inputIndex + resetInputIndex];
                    index += inputSize;
                    super.getLayerSums()[x] = sum;
                    super.getLayerOutput()[x] = sum;
                } else {
                    // if other layer, should still use raw computation.
                    double sum = 0;
                    for(int y = inputIndex; y < limitY; y++) {
                        sum += super.getWeights()[index++] * super.getLayerOutput()[y];
                    }
                    super.getLayerSums()[x] = sum;
                    super.getLayerOutput()[x] = sum;
                }
            }
        }

        super.getActivationFunctions()[currentLayer - 1].activationFunction(super.getLayerOutput(), outputIndex,
                outputSize);

        // update context values
        final int offset = super.getContextTargetOffset()[currentLayer];

        for(int x = 0; x < super.getContextTargetSize()[currentLayer]; x++) {
            super.getLayerOutput()[offset + x] = super.getLayerOutput()[outputIndex + x];
        }
    }

    /**
     * Clone the network.
     * 
     * @return A clone of the network.
     */
    @Override
    public CacheFlatNetwork clone() {
        final CacheFlatNetwork result = new CacheFlatNetwork();
        super.cloneFlatNetwork(result);
        return result;
    }

}
