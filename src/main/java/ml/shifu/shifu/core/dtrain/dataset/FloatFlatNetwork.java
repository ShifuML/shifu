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
 * To solve float input and output types.
 */
public class FloatFlatNetwork extends FlatNetwork implements Cloneable {

    private static final long serialVersionUID = -7208969306860840672L;
    
    public FloatFlatNetwork() {
    }

    public FloatFlatNetwork(final FlatLayer[] layers) {
        super(layers);
    }

    public void compute(float[] input, double[] output) {
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];

//        System.arraycopy(input, 0, getLayerOutput(), sourceIndex, getInputCount());
        for(int i = 0; i < getInputCount(); i++) {
            getLayerOutput()[i + sourceIndex] = input[i] ;
        }

        for (int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        System.arraycopy(getLayerOutput(), 0, output, 0, this.getOutputCount());
    }
    
    public void compute(float[] input, float[] output) {
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            getLayerOutput()[i + sourceIndex] = input[i] ;
        }

        for(int i = getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

//        System.arraycopy(getLayerOutput(), 0, output, 0, this.getOutputCount());
        for(int i = 0; i < getOutputCount(); i++) {
            output[i] = (float)getLayerOutput()[i] ;
        }
    }
    
    /**
     * Clone the network.
     * 
     * @return A clone of the network.
     */
    @Override
    public FloatFlatNetwork clone() {
        final FloatFlatNetwork result = new FloatFlatNetwork();
        super.cloneFlatNetwork(result);
        return result;
    }

}
