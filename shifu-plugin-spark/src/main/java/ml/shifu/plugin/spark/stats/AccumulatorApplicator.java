package ml.shifu.plugin.spark.stats;

import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;

import org.apache.spark.Accumulable;
import org.apache.spark.api.java.function.VoidFunction;

/*
 * The transform which takes a Accumulable object and only accumulates over an RDD
 * using the object.
 */
public class AccumulatorApplicator implements VoidFunction<String> {

    Accumulable<ColumnStateArray, String> accum;
    
    AccumulatorApplicator(Accumulable<ColumnStateArray, String> accum) {
        this.accum= accum;
    }
    
    public void call(String line) throws Exception {
        accum.add(line);
    }

}
