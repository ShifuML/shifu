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
package ml.shifu.shifu.core.dtrain.wnd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import ml.shifu.guagua.io.Bytable;

/**
 * @author juguo
 */
public abstract class AbstractLayer<FIN, FOUT, BIN, BOUT> implements Layer<FIN, FOUT, BIN, BOUT>, Bytable {
    /**
     * Serialization type.
     * In convention, this field should only be serialized in {@link ml.shifu.shifu.core.dtrain.wnd.WideAndDeep}.
     * {@link ml.shifu.shifu.core.dtrain.wnd.Layer} should not serialize this field so as to save space.
     */
    protected SerializationType serializationType = SerializationType.MODEL_SPEC;

    /**
     * @return the serializationType
     */
    public SerializationType getSerializationType() {
        return serializationType;
    }

    /**
     * @param serializationType
     *            the serializationType to set
     */
    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    /**
     * Serialize layer based on provided SerializationType. The implementation of {@link #write(DataOutput)} should use
     * {@link #serializationType} to determine what to serialize.
     * 
     * @param out
     * @param serializationType
     * @throws IOException
     */
    public void write(DataOutput out, SerializationType serializationType) throws IOException {
        setSerializationType(serializationType);
        write(out);
    }

    /**
     * De-serialize layer based on provided SerializationType. The implementation of {@link #readFields(DataInput)}
     * should use {@link #serializationType} to determine what to de-serialize.
     * 
     * @param in
     * @param serializationType
     * @throws IOException
     */
    public void readFields(DataInput in, SerializationType serializationType) throws IOException {
        setSerializationType(serializationType);
        readFields(in);
    }

}

enum SerializationType {
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
