/*
 * Copyright [2013-2015] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.evaluation;

import java.util.List;

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Class for computing area under curve.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public class AreaUnderCurve {
    
    /**
     * Calculate trapezoid area under two points.
     * 
     * @param first the point with the lower X coordinate.
     * @param second the point with the higher X coordinate.
     * @return trapezoid area
     */
    public static double trapezoid(double[] first, double[] second) {
        return (second[1] + first[1]) * (second[0] - first[0]) / 2.0;
    }
    
    
    /**
     * Calculate area of ROC curve based on the PerformanceObject List.
     * 
     * @param curves PerformanceObject List contains ROC curve data.
     * @return area under ROC.
     */
    public static double ofRoc(List<PerformanceObject> roc) {
        return calculate(CurveIteratorFactory.getRocIterator(roc));
    }
    
    /**
     * Calculate area of ROC curve based on the PerformanceObject List.
     * 
     * @param curves PerformanceObject List contains ROC curve data.
     * @return area under ROC.
     */
    public static double ofWeightedRoc(List<PerformanceObject> weightedRoc) {
        return calculate(CurveIteratorFactory.getWeightedRocIterator(weightedRoc));
    }
    
    /**
     * Calculate area of PR curve based on the PerformanceObject List.
     * 
     * @param curves PerformanceObject List contains PR curve data.
     * @return area under PR.
     */
    public static double ofPr(List<PerformanceObject> pr) {
        return calculate(CurveIteratorFactory.getPrIterator(pr));
    }
    
    /**
     * Calculate area of PR curve based on the PerformanceObject List.
     * 
     * @param curves PerformanceObject List contains PR curve data.
     * @return area under PR.
     */
    public static double ofWeightedPr(List<PerformanceObject> weightedPr) {
        return calculate(CurveIteratorFactory.getWeightedPrIterator(weightedPr));
    }
    
    /**
     * Calculate area of the given curve.
     * 
     * @param curve curve with iterator @see {@link CurveIterator}
     * @return area under curve.
     */
    public static double calculate(CurveIterator curve) {
        if(curve.getPointNum() < 2) {
            throw new IllegalArgumentException("We need at least 2 point to calculate area.");
        }
        
        // accumulate the trapezoid area of every successive two points in the curve.
        double sum = 0.0;
        double[] firstPoint = curve.next();
        double[] secondPoint = null;
        while(curve.hasNext()) {
            secondPoint = curve.next();
            sum += trapezoid(firstPoint, secondPoint);
            firstPoint = secondPoint;
        }
        
        return sum;
    }
    
    
}
