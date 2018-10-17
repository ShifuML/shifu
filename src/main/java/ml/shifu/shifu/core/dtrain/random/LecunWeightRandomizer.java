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

import org.encog.neural.networks.BasicNetwork;

/**
 * Copied from https://github.com/tensorflow/tensorflow/blob/63d75cf3332aa2d4e9dd487f1ebfb489756d334b/tensorflow/python/ops/init_ops.py#L1211.
 */
public class LecunWeightRandomizer extends AbstractWeightRandomizer {
    public LecunWeightRandomizer() {
        super(System.currentTimeMillis());
    }

    public LecunWeightRandomizer(long seed) {
        super(seed);
    }
    
    /**
     * Randomize one level of a neural network.
     * 
     * Reference: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
     * 
     *  It draws samples from a uniform distribution within [-limit, limit]
     *  where `limit` is `sqrt(3 / fan_in)`
     *  where `fromCount` is the number of input units in the weight tensor
     * 
     * @param network
     *            The network to randomize
     * @param fromLayer
     *            The from level to randomize.
     */
    public void randomize(final BasicNetwork network, final int fromLayer) {
        final int fromCount = network.getLayerNeuronCount(fromLayer);
        final int toCount = network.getLayerNeuronCount(fromLayer + 1);

        double limit =  Math.sqrt(3.0 / ((double)fromCount));
        
        for(int fromNeuron = 0; fromNeuron < fromCount; fromNeuron++) {
            // biases
            for(int toNeuron = 0; toNeuron < toCount; toNeuron++) {
                network.setWeight(fromLayer, fromCount, toNeuron, 0);
            }

            // weights
            for(int toNeuron = 0; toNeuron < toCount; toNeuron++) {
                double w = super.rnd.nextDouble(-limit, limit);
                network.setWeight(fromLayer, fromNeuron, toNeuron, w);
            }
        }
    }
}