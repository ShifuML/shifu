package ml.shifu.shifu.core.evaluation;

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Class for extracting performance from PerformanceObject.
 * You can find some useful implementation in {@link PerformanceExtractors}.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public interface PerformanceExtractor<T> {
    T extract(PerformanceObject perform);
}
