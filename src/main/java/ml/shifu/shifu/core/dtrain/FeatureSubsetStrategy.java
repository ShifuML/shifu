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
package ml.shifu.shifu.core.dtrain;

/**
 * In each iteration, each node, how many features should be collected from workers.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public enum FeatureSubsetStrategy {

    ALL /* All features */, 
    HALF /* A half features */, 
    ONETHIRD /* One third features */, 
    TWOTHIRDS /* Two third features */, 
    AUTO, /* tree num = 1, ALL, else ONETHIRD */
    SQRT, /* math.sqrt features */
    LOG2; /* math.log2 features */
    
    /**
     * Get {@link FeatureSubsetStrategy} by string, case can be ignored.
     * 
     * @param strategy
     *            the stragy
     * @return the strategy of {@link FeatureSubsetStrategy}
     */
    public static FeatureSubsetStrategy of(String strategy) {
        for(FeatureSubsetStrategy element: values()) {
            if(element.toString().equalsIgnoreCase(strategy)) {
                return element;
            }
        }
        throw new IllegalArgumentException("cannot find such enum in FeatureSubsetStrategy");
    }
}
