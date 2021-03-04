/*
 * Copyright [2013-2020] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain;

/**
 * In each record of training, default weight is 1 while sometimes weighted column like dollar in u-to-u transaction to
 * set. {@link WeightPolicy} is to set different weight policy like only set weight in positive record or no weight set.
 */
public enum WeightPolicy {

    RAW { // RAW weight policy (by default)
        @Override
        public float weight(boolean isPositive, float rawWgt) {
            return rawWgt;
        }
    },
    POSITIVE { // Only apply weight for positive records.
        @Override
        public float weight(boolean isPositive, float rawWgt) {
            return isPositive ? rawWgt : 1f;
        }
    },
    NO { // Apply no weight, default 1
        @Override
        public float weight(boolean isPositive, float rawWgt) {
            return 1f;
        }
    };

    public static WeightPolicy of(String wpStr) {
        for(WeightPolicy wp: values()) {
            if(wp.toString().equalsIgnoreCase(wpStr)) {
                return wp;
            }
        }
        throw new IllegalArgumentException("Cannot find enum with String " + wpStr);
    }

    public abstract float weight(boolean isPositive, float rawWgt);

}
