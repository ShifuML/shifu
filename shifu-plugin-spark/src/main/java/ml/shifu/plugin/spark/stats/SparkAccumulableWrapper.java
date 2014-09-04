/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.plugin.spark.stats;

import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;

import org.apache.spark.AccumulableParam;

/**
 * This is a wrapper class for ColumnStateArray required by Spark.
 */
public class SparkAccumulableWrapper implements
        AccumulableParam<ColumnStateArray, String> {


    public ColumnStateArray addAccumulator(ColumnStateArray stateArray, String row) {
        stateArray.addData(row);
        return stateArray;
    }

    public ColumnStateArray addInPlace(ColumnStateArray stateArray1, ColumnStateArray stateArray2) {
        try {
            return stateArray1.merge(stateArray2);
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        return stateArray1;
    }

    public ColumnStateArray zero(ColumnStateArray initValue) {
        try {
            return initValue.getNewBlank();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

}
