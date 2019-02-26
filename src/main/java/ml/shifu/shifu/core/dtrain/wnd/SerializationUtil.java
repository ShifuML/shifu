package ml.shifu.shifu.core.dtrain.wnd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class SerializationUtil {

    /**
     * Serialize float array with null and length check.
     * 
     * @param out
     * @param array
     * @param size
     * @throws IOException
     */
    public static void writeFloatArray(DataOutput out, float[] array, int size) throws IOException {
        if(array == null || array.length != size) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            for(float value: array) {
                out.writeFloat(value);
            }
        }
    }

    /**
     * De-serialize float array. Will try to use provided array to save memory, but will re-create array if provided is
     * null or size dose not match.
     * 
     * @param in
     * @param array
     * @param size
     * @return
     * @throws IOException
     */
    public static float[] readFloatArray(DataInput in, float[] array, int size) throws IOException {
        if(in.readBoolean()) {
            if(array == null || array.length != size) {
                array = new float[size];
            }
            for(int i = 0; i < size; i++) {
                array[i] = in.readFloat();
            }
        }
        return array;
    }

    /**
     * Serialize two dimensional float array with null and length check.
     * 
     * @param out
     * @param array
     * @param width
     * @param length
     * @throws IOException
     */
    public static void write2DimFloatArray(DataOutput out, float[][] array, int width, int length) throws IOException {
        if(array == null || array.length != width || array[0].length != length) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            for(int i = 0; i < width; i++) {
                for(int j = 0; j < length; j++) {
                    out.writeFloat(array[i][j]);
                }
            }
        }
    }

    /**
     * De-serialize two dimensional float array. Will try to use provided array to save memory, but will re-create array
     * if provided is null or size does not match.
     * 
     * @param in
     * @param array
     * @param width
     * @param length
     * @return
     * @throws IOException
     */
    public static float[][] read2DimFloatArray(DataInput in, float[][] array, int width, int length)
            throws IOException {
        if(in.readBoolean()) {
            if(array == null || array.length != width || array[0].length != length) {
                array = new float[width][length];
            }
            for(int i = 0; i < width; i++) {
                for(int j = 0; j < length; j++) {
                    array[i][j] = in.readFloat();
                }
            }
        }
        return array;
    }

}
