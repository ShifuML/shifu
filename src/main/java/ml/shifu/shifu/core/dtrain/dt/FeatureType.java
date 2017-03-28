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

    CONTINUOUS((byte) 1), CATEGORICAL((byte) 2);

    /**
     * byte type for saving space
     */
    private final byte byteType;

    private FeatureType(byte byteType) {
        this.byteType = byteType;
    }

    public boolean isNumerical() {
        return byteType == CONTINUOUS.getByteType();
    }

    public boolean isCategorical() {
        return byteType == CATEGORICAL.getByteType();
    }

    public static FeatureType of(String featureType) {
        for(FeatureType ft: values()) {
            if(ft.toString().equalsIgnoreCase(featureType)) {
                return ft;
            }
        }
        throw new IllegalArgumentException("Cannot find FeatureType " + featureType);
    }

    public static FeatureType of(byte byteType) {
        for(FeatureType ft: values()) {
            if(ft.getByteType() == byteType) {
                return ft;
            }
        }
        throw new IllegalArgumentException("Cannot find byte of FeatureType for " + byteType);
    }

    public byte getByteType() {
        return byteType;
    }

}
