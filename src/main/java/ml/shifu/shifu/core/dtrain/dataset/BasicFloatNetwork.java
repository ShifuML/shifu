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

import java.lang.reflect.Field;
import java.util.Set;

import ml.shifu.shifu.util.ClassUtils;

import org.encog.ml.data.MLData;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.networks.BasicNetwork;

/**
 * Extends {@link BasicNetwork} to support {@link FloatNeuralStructure} and to embed {@link FloatFlatNetwork}.
 */
public class BasicFloatNetwork extends BasicNetwork {

    private static final long serialVersionUID = -3706149180110298454L;

    private Set<Integer> featureSet;

    public BasicFloatNetwork() {
        Field field = ClassUtils.getDeclaredFieldIncludeSuper("structure", getClass());
        field.setAccessible(true);
        try {
            field.set(this, new FloatNeuralStructure(this));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return the featureSet
     */
    public Set<Integer> getFeatureSet() {
        return featureSet;
    }

    /**
     * @param featureSet
     *            the featureSet to set
     */
    public void setFeatureSet(Set<Integer> featureSet) {
        this.featureSet = featureSet;
    }

    /**
     * Get the layer output. Should be called after {@link #compute(MLData)} or {@link #compute(double[], double[])}.
     * 
     * @param layer
     *            The layer.
     * @return The output from the last call to compute.
     */
    public final double[] getLayerOutput(final int layer) {
        super.getStructure().requireFlat();
        final int layerNumber = getLayerCount() - layer - 1;
        int layerCount = super.getStructure().getFlat().getLayerCounts()[layerNumber];

        int index = super.getStructure().getFlat().getLayerIndex()[layerNumber];
        final double[] output = super.getStructure().getFlat().getLayerOutput();

        double[] results = new double[layerCount];
        for(int i = 0; i < layerCount; i++) {
            if(index >= output.length) {
                throw new NeuralNetworkError("The layer index: " + index
                        + " specifies an output index larger than the network has.");
            }

            results[i] = output[index];
            index += 1;
        }
        return results;
    }

}
