package ml.shifu.shifu.udf.stats;

import java.util.List;

/**
 * Created by Mark on 5/27/2016.
 */
public abstract class Counter<T> {

    public abstract void addData(T val);
    public abstract List<Integer> getCounter();
}
