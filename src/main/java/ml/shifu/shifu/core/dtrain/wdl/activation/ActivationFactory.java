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
package ml.shifu.shifu.core.dtrain.wdl.activation;

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
    private static final Activation DEFAULT_ACTIVATION = new ReLU();

    private static ActivationFactory activationFactory;

    private static Map<String, Activation> actionList = new HashMap<String, Activation>() {
        private static final long serialVersionUID = -7080829888400897248L;
        {
            Reflections reflections = new Reflections("ml.shifu.shifu.core.dtrain.wdl");
            Set<Class<? extends Activation>> classes = reflections.getSubTypesOf(Activation.class);
            for(Class<? extends Activation> activation: classes) {
                try {
                    put(activation.getName().toLowerCase(), activation.newInstance());
                } catch (InstantiationException e) {
                    LOG.error("Don't have empty construction method for " + activation.getName(), e);
                } catch (IllegalAccessException e) {
                    LOG.error("Don't have public construction method for " + activation.getName(), e);
                }
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
     *         Activation if matched, else {@link #DEFAULT_ACTIVATION}
     */
    public Activation getActivation(String name) {
        if(name == null) {
            LOG.error("Input activation name is null, return default activation " + DEFAULT_ACTIVATION);
            return DEFAULT_ACTIVATION;
        }
        return actionList.getOrDefault(name.trim().toLowerCase(), DEFAULT_ACTIVATION);
    }

}
