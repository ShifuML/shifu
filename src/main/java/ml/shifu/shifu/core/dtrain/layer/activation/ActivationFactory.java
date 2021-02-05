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
package ml.shifu.shifu.core.dtrain.layer.activation;

import org.reflections.Reflections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Singleton instance to get Activation from the activation name.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class ActivationFactory {

    private static final Logger LOG = LoggerFactory.getLogger(ActivationFactory.class);

    private static ActivationFactory activationFactory;

    private static Map<String, Class<? extends Activation>> actionList = new HashMap<String, Class<? extends Activation>>() {

        private static final long serialVersionUID = -7080829888400897248L;
        {
            Reflections reflections = new Reflections("ml.shifu.shifu.core");
            Set<Class<? extends Activation>> classes = reflections.getSubTypesOf(Activation.class);
            for(Class<? extends Activation> activation: classes) {
                put(activation.getSimpleName().toLowerCase(), activation);
            }
        }
    };

    private ActivationFactory() {
        LOG.info(actionList.size() + " type activation loaded into system.");
    }

    /**
     * Public method to get instance
     *
     * @return
     *         The singleton instance.
     */
    public static ActivationFactory getInstance() {
        if(activationFactory == null) {
            activationFactory = new ActivationFactory();
        }
        return activationFactory;
    }

    /**
     * Get Activation by the name.
     *
     * @param name
     *            the activation name.
     * @return
     *         Activation if matched, else {@link #newDefaultActiveFunction()}
     */
    public Activation getActivation(String name) {
        if(name == null || actionList.get(name.trim().toLowerCase()) == null) {
            LOG.error("Input activation name is null, return default activation");
            return newDefaultActiveFunction();
        }
        try {
            return actionList.get(name.trim().toLowerCase()).newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            LOG.error("Exception when create new activation, return default activation", e);
            return newDefaultActiveFunction();
        }
    }

    private Activation newDefaultActiveFunction() {
        return new ReLU();
    }
}
