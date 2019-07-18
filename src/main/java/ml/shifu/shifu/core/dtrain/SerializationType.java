package ml.shifu.shifu.core.dtrain;

import java.util.Arrays;

/**
 * @author haillu
 * @date 7/16/2019 4:46 PM
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
