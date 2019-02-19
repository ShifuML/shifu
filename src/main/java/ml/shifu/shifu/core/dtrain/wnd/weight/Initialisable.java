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
package ml.shifu.shifu.core.dtrain.wnd.weight;

/**
 * Class Description.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public interface Initialisable {
    /**
     * Init with one float number.
     *
     * @return the init number.
     */
    float initWeight();

    /**
     * Init a one dimensional float array with specific length
     * @param length, the array length
     * @return the init array
     */
    float[] initWeight(int length);

    /**
     * Init a two dimensional float array with specific row and column
     *
     * @param row the row number
     * @param col the column number
     * @return the init array
     */
    float[][] initWeight(int row, int col);

    /**
     * Convert double number to float
     *
     * @param d the input double value
     * @return the float value converted
     */
    static float getFloat(double d) {
        return Double.valueOf(d).floatValue();
    }

    /**
     * Convert double array to float array
     *
     * @param dArray the double array to convert
     * @param width  the array's width
     * @return the float array after converted
     */
    static float[] getFloat(double[] dArray, int width) {
        float[] weight = new float[width];
        for(int i = 0; i < width; i++) {
            weight[i] = getFloat(dArray[i]);
        }
        return weight;
    }

    /**
     * Convert two dimension double array to float array
     *
     * @param dArray the double array to convert
     * @param row    the row number
     * @param col    the column number
     * @return the float array after converted
     */
    static float[][] getFloat(double[][] dArray, int row, int col) {
        float[][] weight = new float[row][col];
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                weight[i][j] = getFloat(dArray[i][j]);
            }
        }
        return weight;
    }
}
