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
package ml.shifu.shifu.core.dtrain.dt;

/**
 * Two feature types supported in our decision tree algorithm: {@link #CONTINUOUS} and {@link #CATEGORICAL} types.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public enum FeatureType {
    CONTINUOUS, CATEGORICAL;

    public static FeatureType of(String featureType) {
        for(FeatureType ft: values()) {
            if(ft.toString().equalsIgnoreCase(featureType)) {
                return ft;
            }
        }
        throw new IllegalArgumentException("Cannot find FeatureType " + featureType);
    }
}
