package ml.shifu.shifu.core.evaluation;

import java.util.List;

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Class for generate different curve iterator on PerformanceObject List.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public class CurveIteratorFactory {

    public static CurveIterator getRocIterator(List<PerformanceObject> performs) {
        return new RocIterator(performs);
    }
    
    public static CurveIterator getWeightedRocIterator(List<PerformanceObject> performs) {
        return new WeightedRocIterator(performs);
    }
    
    public static CurveIterator getPrIterator(List<PerformanceObject> performs) {
        return new PrIterator(performs);
    }
    
    public static CurveIterator getWeightedPrIterator(List<PerformanceObject> performs) {
        return new WeightedPrIterator(performs);
    }
    
    /**
     * CurveIterator implementation for roc curve iterating.
     * 
     * <p>
     * Use next() to get next point from the given performance list.
     * Returned point is represented by a double array with two elments as (fpr, recall) or 
     * (weightedFpr, weightedRecall) when isWeighted is true.
     * </p>
     */
    private static class RocIterator extends CurveIterator {
        
        public RocIterator(List<PerformanceObject> performs) {
            super(performs);
        }

        @Override
        public double[] next() {
            PerformanceObject perform = super.getNextPerformanceObject();
            return new double[]{perform.fpr, perform.recall};
        }
    }
    
    /**
     * CurveIterator implementation for weighted roc curve iterating.
     * 
     * <p>
     * Use next() to get next point from the given performance list.
     * Returned point is represented by a double array with two elments as (weightedFpr, weightedRecall).
     * </p>
     */
    private static class WeightedRocIterator extends CurveIterator {
        
        public WeightedRocIterator(List<PerformanceObject> performs) {
            super(performs);
        }

        @Override
        public double[] next() {
            PerformanceObject perform = super.getNextPerformanceObject();
            return new double[]{perform.weightedFpr, perform.weightedRecall};
        }
    }
    
    /**
     * CurveIterator implementation for pr curve iterating.
     * 
     * <p>
     * Use next() to get next point from the given performance list.
     * Returned point is represented by a double array with two elments as (recall, precision).
     * </p>
     */
    private static class PrIterator extends CurveIterator {
        
        public PrIterator(List<PerformanceObject> performs) {
            super(performs);
        }
        
        @Override
        public double[] next() {
            PerformanceObject perform = super.getNextPerformanceObject();
            return new double[]{perform.recall, perform.precision};
        }
    }
    
    /**
     * CurveIterator implementation for weighted pr curve iterating.
     * 
     * <p>
     * Use next() to get next point from the given performance list.
     * Returned point is represented by a double array with two elments as (weightedRecall, weightedPrecision).
     * </p>
     */
    private static class WeightedPrIterator extends CurveIterator {
        
        public WeightedPrIterator(List<PerformanceObject> performs) {
            super(performs);
        }
        
        @Override
        public double[] next() {
            PerformanceObject perform = super.getNextPerformanceObject();
            return new double[]{perform.weightedRecall, perform.weightedPrecision};
        }
    }
}
