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
package ml.shifu.shifu.core.dtrain.layer;

import java.util.Arrays;

/**
 * Serialize parameters or gradients from type of {@link SerializationType}.
 * 
 * @author pengzhang
 */
public enum SerializationType {
    /**
     * Serialize types, each of them including different serialize scope
     */
    WEIGHTS(0), GRADIENTS(1), MODEL_SPEC(2), ERROR(-1);

    int value;

    SerializationType(int type) {
        this.value = type;
    }

    public static SerializationType getSerializationType(int value) {
        return Arrays.stream(values()).filter(type -> type.value == value).findFirst().orElse(SerializationType.ERROR);
    }

    public int getValue() {
        return this.value;
    }
}
