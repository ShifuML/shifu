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
package ml.shifu.shifu.core.stability.algorithm;

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * NullValueChaosAlgorithm return null for category value and 0 for numeric value.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class NullValueChaosAlgorithm extends BaseChaosAlgorithm {

    @Override
    String generateCategoryChaosValue(String originValue, ColumnConfig config) {
        return null;
    }

    @Override
    String generateNumericChaosValue(String originValue, ColumnConfig config) {
        return "0";
    }
}
