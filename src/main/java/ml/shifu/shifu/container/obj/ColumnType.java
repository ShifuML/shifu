/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.container.obj;

public enum ColumnType {
    A((byte) 0), N((byte) 1), C((byte) 2), H((byte) 3); // H means hybrid, which is numerical type for major values,
                                                        // while for missing values, it is split into different
                                                        // categories
    /**
     * byte type for saving space
     */
    private final byte byteType;

    private ColumnType(byte byteType) {
        this.byteType = byteType;
    }

    public boolean isNumerical() {
        return byteType == N.getByteType();
    }

    public boolean isCategorical() {
        return byteType == C.getByteType();
    }

    public static ColumnType of(String columnType) {
        for(ColumnType ft: values()) {
            if(ft.toString().equalsIgnoreCase(columnType)) {
                return ft;
            }
        }
        throw new IllegalArgumentException("Cannot find ColumnType " + columnType);
    }

    public static ColumnType of(byte byteType) {
        for(ColumnType ft: values()) {
            if(ft.getByteType() == byteType) {
                return ft;
            }
        }
        throw new IllegalArgumentException("Cannot find byte of ColumnType for " + byteType);
    }

    public byte getByteType() {
        return byteType;
    }
}
