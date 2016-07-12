package ml.shifu.shifu.udf.stats;

import java.util.List;

/**
 * Created by Mark on 5/27/2016.
 */
public abstract class Counter {

    public abstract void addData(String val);
    public abstract List<Long> getCounter();
    public abstract double getUnitMean();
    public abstract double getMissingRate();
    public abstract long getTotalInstCnt();
}
