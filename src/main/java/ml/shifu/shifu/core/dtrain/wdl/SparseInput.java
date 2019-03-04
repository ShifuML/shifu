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

/**
 * {@link SparseInput} is only to save non-zero value instead of one sparse float array to save memory.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class SparseInput {

    /**
     * Column index which is columnNum in ColumnConfig.json
     */
    private int columnIndex;

    /**
     * Value index is index in ColumnConfig#binCategory list which denotes which category this variable is.
     * TODO missing value index? use short to save memory? short only for 65536, 65536 categories?
     */
    private int valueIndex;

    /**
     * By default value is 1 for sparse input while in other senarios, value could not be 1.
     */
    private float value = 1f;

    // TODO if add weight or change valueIndex to array for multiple-hot encoder

    /**
     * Constructor with column index and value index
     * 
     * @param columnIndex
     *            column index in ColumnConfig#columnNum
     * @param valueIndex
     *            value index, which value of such category
     */
    public SparseInput(int columnIndex, int valueIndex) {
        this.columnIndex = columnIndex;
        this.valueIndex = valueIndex;
    }

    /**
     * Constructor with column index, value index and value
     * 
     * @param columnIndex
     *            column index in ColumnConfig#columnNum
     * @param valueIndex
     *            value index, which value of such category
     * @param value
     *            the category value
     */
    public SparseInput(int columnIndex, int valueIndex, float value) {
        this.columnIndex = columnIndex;
        this.valueIndex = valueIndex;
        this.value = value;
    }

    /**
     * @return the columnIndex
     */
    public int getColumnIndex() {
        return columnIndex;
    }

    /**
     * @param columnIndex
     *            the columnIndex to set
     */
    public void setColumnIndex(int columnIndex) {
        this.columnIndex = columnIndex;
    }

    /**
     * @return the valueIndex
     */
    public int getValueIndex() {
        return valueIndex;
    }

    /**
     * @param valueIndex
     *            the valueIndex to set
     */
    public void setValueIndex(int valueIndex) {
        this.valueIndex = valueIndex;
    }

    /**
     * @return the value
     */
    public float getValue() {
        return value;
    }

    /**
     * @param value
     *            the value to set
     */
    public void setValue(float value) {
        this.value = value;
    }

}
