/*
 * This is a wrapper class for ColumnStateArray required by Spark.
 */

package ml.shifu.plugin.spark.stats;

import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;

import org.apache.spark.AccumulableParam;

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
