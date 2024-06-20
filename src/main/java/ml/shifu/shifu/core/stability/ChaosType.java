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

import ml.shifu.shifu.core.stability.algorithm.ChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.DeviationChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.NullValueChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.RandomChaosAlgorithm;

/**
 * Enum define different chaos type. For each chaos type, we should consider different logical for both category and
 * numeric value transfer behaviours.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public enum ChaosType {
    NULL_VALUE("null", new NullValueChaosAlgorithm(), "return null for category value and return"),
    RANDOM_VALUE("random", new RandomChaosAlgorithm(), "return random value for numeric, random choose category value from existing category type"),
    DEVIATION_VALUE("deviation", new DeviationChaosAlgorithm(), "Return deviation value for both category and numeric");

    private String name;
    private String description;
    private ChaosAlgorithm chaosAlgorithm;

    ChaosType(String name, ChaosAlgorithm chaosAlgorithm, String description) {
        this.name = name;
        this.chaosAlgorithm = chaosAlgorithm;
        this.description = description;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public ChaosAlgorithm getChaosAlgorithm() {
        return chaosAlgorithm;
    }

    public static ChaosType fromName(String name) {
        for(ChaosType chaosType: ChaosType.values()) {
            if(chaosType.getName().equalsIgnoreCase(name)) {
                return chaosType;
            }
        }
        return null;
    }
}
