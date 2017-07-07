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

import org.encog.engine.network.activation.ActivationLinear;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.flat.FlatLayer;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.structure.NeuralStructure;

/**
 * Extend {@link NeuralStructure} to set {@link FloatFlatNetwork}.
 * 
 * <p>
 * {@link #finalizeStruct()} is used to replace {@link #finalizeStructure()} as {@link #finalizeStructure()} is set to
 * final and cannot be override.
 */
public class FloatNeuralStructure extends NeuralStructure {

    private static final long serialVersionUID = 7662087479144051670L;

    public FloatNeuralStructure(BasicNetwork network) {
        super(network);
    }

    /**
     * Build the synapse and layer structure. This method should be called afteryou are done adding layers to a network,
     * or change the network's logic property.
     */
    public void finalizeStruct() {
        if(this.getLayers().size() < 2) {
            throw new NeuralNetworkError("There must be at least two layers before the structure is finalized.");
        }

        final FlatLayer[] flatLayers = new FlatLayer[this.getLayers().size()];

        for(int i = 0; i < this.getLayers().size(); i++) {
            final BasicLayer layer = (BasicLayer) this.getLayers().get(i);
            if(layer.getActivation() == null) {
                layer.setActivation(new ActivationLinear());
            }

            flatLayers[i] = layer;
        }

        this.setFlat(new FloatFlatNetwork(flatLayers, true));

        finalizeLimit();
        this.getLayers().clear();
        enforceLimit();
    }

}
