package ml.shifu.shifu.core.evaluation;

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Factory class for getting some useful PerformanceExtractor implementation,
 * includes curve point extractors for ROC, weighted ROC, PR, weighted PR curve.
 * 
 * @see RocPointExtractor
 * @see WeightRocPointExtractor
 * @see PrPointExtractor
 * @see WeightPrPointExtractor
 *      
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public class PerformanceExtractors {
    
    private static RocPointExtractor roc = new RocPointExtractor();
    private static WeightRocPointExtractor wroc = new WeightRocPointExtractor();
    private static PrPointExtractor pr = new PrPointExtractor();
    private static WeightPrPointExtractor wpr = new WeightPrPointExtractor();
    
    public static PerformanceExtractor<double[]> getRocPointExtractor() {
        return roc;
    }
    
    public static PerformanceExtractor<double[]> getWeightRocPointExtractor() {
        return wroc;
    }
    
    public static PerformanceExtractor<double[]> getPrPointExtractor() {
        return pr;
    }
    
    public static PerformanceExtractor<double[]> getWeightPrPointExtractor() {
        return wpr;
    }
}

/**
 * Extractor used to extract point - (fpr,recall) from PerformanceExtractor object.
 * Use double array to represent the point.
 */
class RocPointExtractor implements PerformanceExtractor<double[]> {

    @Override
    public double[] extract(PerformanceObject perform) {
        return new double[]{perform.fpr, perform.recall};
    }
    
}

/**
 * Extractor used to extract point - (weightedFpr,weightedRecall) from PerformanceExtractor object.
 * Use double array to represent the point.
 */
class WeightRocPointExtractor implements PerformanceExtractor<double[]> {
    
    @Override
    public double[] extract(PerformanceObject perform) {
        return new double[]{perform.weightedFpr, perform.weightedRecall};
    }
    
}

/**
 * Extractor used to extract point - (recall,precision) from PerformanceExtractor object.
 * Use double array to represent the point.
 */
class PrPointExtractor implements PerformanceExtractor<double[]> {
    
    @Override
    public double[] extract(PerformanceObject perform) {
        return new double[]{perform.recall, perform.precision};
    }
    
}

/**
 * Extractor used to extract point - (weightedRecall,weightedPrecision) from PerformanceExtractor object.
 * Use double array to represent the point.
 */
class WeightPrPointExtractor implements PerformanceExtractor<double[]> {
    
    @Override
    public double[] extract(PerformanceObject perform) {
        return new double[]{perform.weightedRecall, perform.weightedPrecision};
    }
    
}
