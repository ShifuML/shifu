/*
 * Copyright [2013-2021] PayPal Software Foundation
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
package ml.shifu.shifu.core.stability;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.stability.algorithm.BaseChaosAlgorithm;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.reflections.Reflections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * ChaosAlgorithmFactory to get chaos factory class by it's name.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class ChaosFactory {
    private static final Logger LOG = LoggerFactory.getLogger(ChaosFactory.class);
    private ChaosType chaosType = ChaosType.fromName(Environment.getProperty(Constants.CHAOS_TYPE));
    private Set<String> chaosColumnSet = new HashSet<>(Arrays.asList(Environment.getProperty(Constants.CHAOS_COLUMNS).split(",")));

    private ChaosFactory() {
        LOG.info("ChaosFactory init with chaos type {}: {}", this.chaosType.getName(), this.chaosType.getDescription());
        LOG.info("Inject chaos type on columns: {}", Environment.getProperty(Constants.CHAOS_COLUMNS));
    }

    private static class SingletonHolder {
        private static ChaosFactory instance = new ChaosFactory();
    }

    /**
     * Public method to get instance
     *
     * @return
     *         The singleton instance.
     */
    public static ChaosFactory getInstance() {
        return SingletonHolder.instance;
    }

    private static Map<String, Class<? extends BaseChaosAlgorithm>> algorithmMap = new HashMap<String, Class<? extends BaseChaosAlgorithm>>() {
        private static final long serialVersionUID = -1080829888400897248L;
        {
            Reflections reflections = new Reflections("ml.shifu.shifu.core.stability.algorithm");
            Set<Class<? extends BaseChaosAlgorithm>> classes = reflections.getSubTypesOf(BaseChaosAlgorithm.class);
            for(Class<? extends BaseChaosAlgorithm> algorithm: classes) {
                put(algorithm.getName().toLowerCase(), algorithm);
            }
        }
    };

    public ChaosType getChaosType() {
        return this.chaosType;
    }

    public boolean needInjectChaos(ColumnConfig config) {
        LOG.info("Inject chaos for {} use type {}", config.getColumnName(), this.chaosType.getName());
        return this.chaosColumnSet.contains(config.getColumnName().toLowerCase());
    }
}
