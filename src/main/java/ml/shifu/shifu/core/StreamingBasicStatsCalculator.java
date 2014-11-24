package ml.shifu.shifu.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StreamingBasicStatsCalculator {
    private static Logger log = LoggerFactory.getLogger(BasicStatsCalculator.class);

    private Double sum;
    private Double squaredSum;
    private Double min = Double.MAX_VALUE;
    private Double max = Double.MIN_VALUE;
    private Double mean = Double.NaN;
    private Double stdDev = Double.NaN;
    private Double median = Double.NaN;
    private Long varSize = 0l;

    private Double threshold = 1e6;
    private Double EPS = 1e-6;

    private Estimator<Double> medianEstimator;

    public StreamingBasicStatsCalculator() {
        this.sum        = 0.0;
        this.squaredSum = 0.0;
        this.min        = Double.MAX_VALUE;
        this.max        = Double.MIN_VALUE;
        
        //min, median, max
        this.medianEstimator = new Estimator<Double>(3);
    }

    public void aggregate(Double value) {
        if (value.isInfinite() || value.isNaN() || Math.abs(value) > this.threshold) return;

        this.sum       += value;
        this.squaredSum = this.squaredSum + value * value;

        this.max = Math.max(max, value);
        this.min = Math.min(min, value);
        this.varSize ++;

        this.medianEstimator.add(value);

    }

    public void complete() {
        mean = sum / varSize;
        stdDev = Math.sqrt((squaredSum - (sum * sum) / varSize + EPS)
                / (varSize - 1));

        median = medianEstimator.getBin().get(1);
    }

    public Double getMin() {
        return min;
    }

    public Double getMax() {
        return max;
    }

    public Double getMean() {
        return mean;
    }

    public Double getStdDev() {
        return stdDev;
    }

    public Double getMedian() {
        return median;
    }
}
