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
package ml.shifu.shifu.core.dtrain.wdl;

import ml.shifu.shifu.core.dtrain.random.HeWeightRandomizer;
import ml.shifu.shifu.core.dtrain.random.LecunWeightRandomizer;
import ml.shifu.shifu.core.dtrain.random.XavierWeightRandomizer;
import ml.shifu.shifu.core.dtrain.wdl.weight.Initialisable;
import ml.shifu.shifu.core.dtrain.wdl.weight.RangeRandom;
import ml.shifu.shifu.core.dtrain.wdl.weight.WeightRandom;
import ml.shifu.shifu.core.dtrain.wdl.weight.Zero;

/**
 * Class implement this interface should have a method initWeight.
 * <p>
 *
 * @author : Wu Devin (haifwu@paypal.com)
 */
public interface WeightInitializer<SELF extends WeightInitializer> {
    enum InitMethod {
        /**
         * Init all weight with {@link Zero}
         */
        ZERO(new Zero()),
        /**
         * Init all weight with random method define in {@link HeWeightRandomizer}
         */
        HE_RANDOM(new WeightRandom(new HeWeightRandomizer())),
        /**
         * Init all weight with random method define in {@link LecunWeightRandomizer}
         */
        LECUN_RANDOM(new WeightRandom(new LecunWeightRandomizer())),
        /**
         * Init all weight with random method define in {@link XavierWeightRandomizer}
         */
        XAVIER_RANDOM(new WeightRandom(new XavierWeightRandomizer())),
        /**
         * Init all Weight use {@link RangeRandom}
         */
        ZERO_ONE_RANGE_RANDOM(new RangeRandom(0, 1));

        private Initialisable initialisable;

        InitMethod(Initialisable initialisable) {
            this.initialisable = initialisable;
        }

        /**
         * @return the initialisable
         */
        public Initialisable getInitialisable() {
            return initialisable;
        }

        /**
         * Get InitMethod from method name
         * @param method the method name
         * @return InitMethod
         */
        public static InitMethod getInitMethod(String method) {
            InitMethod defaultMethod = ZERO;
            if(method == null) {
                return defaultMethod;
            }
            for(InitMethod m : InitMethod.values()) {
                if(m.name().toLowerCase().equals(method.toLowerCase())) {
                    return m;
                }
            }
            return defaultMethod;
        }
    }

    /**
     * Init weight according to the method
     *
     * @param method, the init method
     */
    void initWeight(InitMethod method);

    /**
     * init weight according to an existing model
     *
     * @param updateModel model to copy weight from
     */
    void initWeight(SELF updateModel);

}