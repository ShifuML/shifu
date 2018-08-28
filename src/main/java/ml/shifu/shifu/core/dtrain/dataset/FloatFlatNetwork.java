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

import java.util.Set;

import org.encog.neural.flat.FlatLayer;
import org.encog.neural.flat.FlatNetwork;

import ml.shifu.shifu.core.dtrain.nn.BasicDropoutLayer;

/**
 * To solve float input and output types.
 */
public class FloatFlatNetwork extends FlatNetwork implements Cloneable {

    private static final long serialVersionUID = -7208969306860840672L;

    /**
     * The dropout rate for each layer.
     */
    private double[] layerDropoutRates;

    public FloatFlatNetwork() {
        this.layerDropoutRates = new double[0];
    }

    public FloatFlatNetwork(final FlatLayer[] layers) {
        this(layers, true);
    }

    public FloatFlatNetwork(final FlatLayer[] layers, boolean dropout) {
        init(layers, dropout);
    }

    private void init(FlatLayer[] layers, boolean dropout) {
        super.init(layers);

        final int layerCount = layers.length;
        if(dropout) {
            this.setLayerDropoutRates(new double[layerCount]);
        } else {
            this.setLayerDropoutRates(new double[0]);
        }

        int index = 0;
        for(int i = layers.length - 1; i >= 0; i--) {
            final FlatLayer layer = layers[i];
            if(dropout && layer instanceof BasicDropoutLayer) {
                this.getLayerDropoutRates()[index] = ((BasicDropoutLayer) layer).getDropout();
            }
            index += 1;
        }
    }

    public void compute(float[] input, double[] output) {
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];

        for(int i = 0; i < getInputCount(); i++) {
            getLayerOutput()[i + sourceIndex] = input[i];
        }

        for(int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        System.arraycopy(getLayerOutput(), 0, output, 0, this.getOutputCount());
    }

    public void compute(float[] input, double[] output, Set<Integer> dropoutNodes) {
    	
        final int sourceIndex = getLayerOutput().length - getLayerCounts()[getLayerCounts().length - 1];
        boolean inputLayerDropoutEnable = isDropoutEnable(getLayerCounts().length - 1, dropoutNodes);
        double nonDropoutRate = (1d - this.getLayerDropoutRates()[getLayerCounts().length - 1]);
        for(int i = 0; i < getInputCount(); i++) {
        	if (inputLayerDropoutEnable) {
        		if (dropoutNodes.contains(i + sourceIndex)) {
        			getLayerOutput()[i + sourceIndex] = 0d;
        		} else {
        		    //To rescale output of each node. Since the total input of next layer reduce when using drop out, we need to make it up.
        			getLayerOutput()[i + sourceIndex] = input[i] / nonDropoutRate;
        		}
        	} else {
        		getLayerOutput()[i + sourceIndex] = input[i];
        	}
        }

        for(int i = this.getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i, dropoutNodes);
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
            getLayerOutput()[i + sourceIndex] = input[i];
        }

        for(int i = getLayerIndex().length - 1; i > 0; i--) {
            computeLayer(i);
        }

        // update context values
        final int offset = getContextTargetOffset()[0];

        for(int x = 0; x < getContextTargetSize()[0]; x++) {
            this.getLayerOutput()[offset + x] = this.getLayerOutput()[x];
        }

        // copy to float output array
        for(int i = 0; i < getOutputCount(); i++) {
            output[i] = (float) getLayerOutput()[i];
        }
    }

    @Override
    protected void computeLayer(final int currentLayer) {
        final int inputIndex = super.getLayerIndex()[currentLayer];
        final int outputIndex = super.getLayerIndex()[currentLayer - 1];
        final int inputSize = super.getLayerCounts()[currentLayer];
        final int outputSize = super.getLayerFeedCounts()[currentLayer - 1];

        int index = super.getWeightIndex()[currentLayer - 1];

        final int limitX = outputIndex + outputSize;
        final int limitY = inputIndex + inputSize;

		// weight values
		for (int x = outputIndex; x < limitX; x++) {
			double sum = 0;
			for (int y = inputIndex; y < limitY; y++) {
				sum += super.getWeights()[index++] * super.getLayerOutput()[y];
			}
			super.getLayerSums()[x] = sum;
			super.getLayerOutput()[x] = sum;
		}
        
        super.getActivationFunctions()[currentLayer - 1].activationFunction(super.getLayerOutput(), outputIndex,
                outputSize);

        // update context values
        final int offset = super.getContextTargetOffset()[currentLayer];

        for(int x = 0; x < super.getContextTargetSize()[currentLayer]; x++) {
            super.getLayerOutput()[offset + x] = super.getLayerOutput()[outputIndex + x];
        }
    }

    protected void computeLayer(final int currentLayer, Set<Integer> dropoutNodes) {   	
        final int inputIndex = super.getLayerIndex()[currentLayer];
        final int outputIndex = super.getLayerIndex()[currentLayer - 1];
        final int inputSize = super.getLayerCounts()[currentLayer];
        final int outputSize = super.getLayerFeedCounts()[currentLayer - 1];
        
        int index = super.getWeightIndex()[currentLayer - 1];

        final int limitX = outputIndex + outputSize;
        final int limitY = inputIndex + inputSize;

		// weight values
		for (int x = outputIndex; x < limitX; x++) {
			double sum = 0;
			for (int y = inputIndex; y < limitY; y++) {
				sum += super.getWeights()[index++] * super.getLayerOutput()[y];
			}
			super.getLayerSums()[x] = sum;
			super.getLayerOutput()[x] = sum;
		}


        super.getActivationFunctions()[currentLayer - 1].activationFunction(super.getLayerOutput(), outputIndex,
                outputSize);

        // if current output layer does not enable dropout, we use normal computelayer method
        if (isDropoutEnable(currentLayer - 1, dropoutNodes)) {
            // dropout nodes' output and rescale remain nodes' output
            final double nonDropoutRate = 1d - this.getLayerDropoutRates()[currentLayer - 1];
            for (int x = outputIndex; x < limitX; x++) {
                if (dropoutNodes.contains(x)) {
                    super.getLayerOutput()[x] = 0d;
                } else {
                    super.getLayerOutput()[x] /= nonDropoutRate;
                }
            }
        }

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
    public FloatFlatNetwork clone() {
        final FloatFlatNetwork result = new FloatFlatNetwork();
        super.cloneFlatNetwork(result);
        return result;
    }

    /**
     * @return the layerDropoutRates
     */
    public double[] getLayerDropoutRates() {
        return layerDropoutRates;
    }

    /**
     * @param layerDropoutRates
     *            the layerDropoutRates to set
     */
    public void setLayerDropoutRates(double[] layerDropoutRates) {
        this.layerDropoutRates = layerDropoutRates;
    }

    public boolean isDropoutEnable(int layer, Set<Integer> dropoutNodes) {
    	return dropoutNodes != null &&
    	        (this.getLayerDropoutRates().length > layer) && 
    			(Double.compare(this.getLayerDropoutRates()[layer], 0d) > 0);
    }
}
