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
package ml.shifu.shifu.core.dtrain.wdl;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SerializationUtil {

    /**
     *  Write false when null
     */
    public static final int NULL = 0;

    /**
     * Serialize double array with null and length check.
     * 
     * @param out
     *            the data output stream
     * @param array
     *            the array to be serialized
     * @param size
     *            the expected length of array
     * @throws IOException
     *             if an I/O error occurs.
     */
    public static void writeDoubleArray(DataOutput out, double[] array, int size) throws IOException {
        if(array == null || array.length != size) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            for(double value: array) {
                out.writeDouble(value);
            }
        }
    }

    /**
     * De-serialize double array. Will try to use provided array to save memory, but will re-create array if provided is
     * null or size dose not match.
     * 
     * @param in
     *            the input stream
     * @param array
     *            the array to hold data from DataInput. Will be ignored if null or length not match size param.
     * @param size
     *            the expected length of array
     * @return de-serialized array. The returned value will reuse memory of provided array if it is not null and its
     *         length is size.
     * @throws IOException
     *             if an I/O error occurs.
     */
    public static double[] readDoubleArray(DataInput in, double[] array, int size) throws IOException {
        if(in.readBoolean()) {
            if(array == null || array.length != size) {
                array = new double[size];
            }
            for(int i = 0; i < size; i++) {
                array[i] = in.readDouble();
            }
        }
        return array;
    }

    /**
     * Serialize two dimensional double array with null and length check.
     * 
     * @param out
     *            the data output stream
     * @param array
     *            the array to be serialized. The size should match [width][length]. The array will be treated as null
     *            otherwise.
     * @param width
     *            array width
     * @param length
     *            array length
     * @throws IOException
     *             if an I/O error occurs.
     */
    public static void write2DimDoubleArray(DataOutput out, double[][] array, int width, int length) throws IOException {
        if(array == null || array.length != width || array[0].length != length) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            for(int i = 0; i < width; i++) {
                for(int j = 0; j < length; j++) {
                    out.writeDouble(array[i][j]);
                }
            }
        }
    }

    /**
     * De-serialize two dimensional double array. Will try to use provided array to save memory, but will re-create array
     * if provided is null or size does not match.
     * 
     * @param in
     *            the data input stream
     * @param array
     *            the array to hold de-serialized data.Will be ignored if null or not sized as [width][length].
     * @param width
     *            array width
     * @param length
     *            array length
     * @return de-serialized 2-dim array. The returned value will reuse memory of provided array if it is not null and
     *         size match with [width][length].
     * @throws IOException
     *             if an I/O error occurs.
     */
    public static double[][] read2DimDoubleArray(DataInput in, double[][] array, int width, int length)
            throws IOException {
        if(in.readBoolean()) {
            if(array == null || array.length != width || array[0].length != length) {
                array = new double[width][length];
            }
            for(int i = 0; i < width; i++) {
                for(int j = 0; j < length; j++) {
                    array[i][j] = in.readDouble();
                }
            }
        }
        return array;
    }

    /**
     * Serialize Integer list with null check.
     * 
     * @param out
     *            the data output stream
     * @param list
     *            the List to serialize.
     * @throws IOException
     *             if an I/O error occurs.
     */
    public static void writeIntList(DataOutput out, List<Integer> list) throws IOException {
        if(list == null) {
            out.writeInt(0);
        } else {
            out.writeInt(list.size());
            for(Integer value: list) {
                out.writeInt(value);
            }
        }
    }

    /**
     * De-serialize Integer list. Try using provided list to save memory.
     * 
     * @param in
     *            the data input stream
     * @param list
     *            the list to hold de-serialized data.
     * @return de-serialized list. Will reuse provided list if it is not null.
     * @throws IOException
     *             if an I/O error occurs.
     */
    public static List<Integer> readIntList(DataInput in, List<Integer> list) throws IOException {
        int size = in.readInt();
        if(size > 0) {
            if(list == null) {
                list = new ArrayList<Integer>();
            } else {
                list.clear();
            }
            for(int i = 0; i < size; i++) {
                list.add(in.readInt());
            }
        }
        return list;
    }

}
