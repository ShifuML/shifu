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
package ml.shifu.shifu.core.dtrain.loss;

/**
 * {@link LossType} defines sqaured loss and binary entropy loss in wide and deep.
 * 
 * @author pengzhang
 */
public enum LossType {

    SQUARED((short) 1), LOG((short) 2);

    private final short type;

    private LossType(short type) {
        this.type = type;
    }

    /**
     * @return the type
     */
    public short getType() {
        return type;
    }

    public static LossType of(String lossType) {
        for(LossType lt: LossType.values()) {
            if(lt.toString().equalsIgnoreCase(lossType)) {
                return lt;
            }
        }
        throw new IllegalArgumentException("Unsupported loss type: " + lossType);
    }

    public static LossType of(short type) {
        for(LossType lt: LossType.values()) {
            if(lt.getType() == type) {
                return lt;
            }
        }
        throw new IllegalArgumentException("Unsupported loss type index: " + type);
    }

}
