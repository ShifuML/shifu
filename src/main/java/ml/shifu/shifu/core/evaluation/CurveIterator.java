package ml.shifu.shifu.core.evaluation;

import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Class for generate cureve point based on the PerformanceObject List.
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public abstract class CurveIterator implements Iterator<double[]>{
    
    private Iterator<PerformanceObject> iter;
    private List<PerformanceObject> performs;
    
    public CurveIterator(List<PerformanceObject> performs) {
        this.performs = performs;
        this.iter = performs.iterator();
    }
    
    /**
     * Reset iterator.
     */
    public void reset() {
        iter = performs.iterator();
    }
    
    public int getPointNum() {
        return performs.size();
    }
    
    protected PerformanceObject getNextPerformanceObject() {
        return iter.next();
    }
    
    public boolean hasNext() {
        return iter.hasNext();
    }
    
    public abstract double[] next();
    
    public void remove() {
        throw new UnsupportedOperationException("You can't delete the curve point");
    };
    
}


