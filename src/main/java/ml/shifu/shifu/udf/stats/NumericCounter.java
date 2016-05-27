package ml.shifu.shifu.udf.stats;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.util.CommonUtils;

/**
 * Created by Mark on 5/27/2016.
 */
public class NumericCounter extends Counter<Object> {

    private List<Double> binBoundary;
    private List<Integer> counter;

    public NumericCounter(List<Double> binBoundary) {
        this.binBoundary = binBoundary;
        this.counter = new ArrayList<Integer>(binBoundary.size() + 1);
    }

    @Override
    public void addData(Object val) {
        Double dVal = ((Number) val).doubleValue();
        int index = CommonUtils.getBinIndex(binBoundary, dVal);
        counter.set(index, counter.get(index) + 1);
    }

    @Override
    public List<Integer> getCounter() {
        return counter;
    }
}
