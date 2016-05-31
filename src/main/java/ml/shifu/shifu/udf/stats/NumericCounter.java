package ml.shifu.shifu.udf.stats;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

import ml.shifu.shifu.util.CommonUtils;

/**
 * Created by Mark on 5/27/2016.
 */
public class NumericCounter extends Counter {

    private final static Logger logger = LoggerFactory.getLogger(NumericCounter.class);
    private List<Double> binBoundary;
    private Integer[] counter;
    private String name;

    public NumericCounter(String name, List<Double> binBoundary) {
        this.name = name;
        this.binBoundary = binBoundary;
        this.counter = new Integer[binBoundary.size() + 1];
        Arrays.fill(counter, 0);
    }

    @Override
    public void addData(Object val) {
        try {
            Double dVal = Double.parseDouble(val.toString());
            int index = CommonUtils.getBinIndex(binBoundary, dVal);
            counter[index] = counter[index] + 1;
        } catch (Exception e) {
            logger.warn(String.format("Unable to logger this column %s with %s", name, val));
            counter[binBoundary.size()] = counter[binBoundary.size()] + 1;
        }
    }

    @Override
    public List<Integer> getCounter() {
        return Arrays.asList(counter);
    }
}
