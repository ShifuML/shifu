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

import java.util.Iterator;
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
     * @param x1 x of first point.
     * @param y1 y of first point.
     * @param x2 x of second point, note: x2 is always considered to be no less than x1, so (x2 - x1) >= 0.
     * @param y2 y of second point.
     * @return trapezoid area
     */
    public static double trapezoid(double x1, double y1, double x2, double y2) {
        return (y2 + y1) * (x2 - x1) / 2.0;
    }
    
    
    /**
     * Calculate area of ROC curve based on the PerformanceObject List.
     * 
     * @param roc PerformanceObject List contains ROC curve data.
     * @return area under ROC.
     */
    public static double ofRoc(List<PerformanceObject> roc) {
        return calculateArea(roc, Performances.getFprExtractor(), Performances.getRecallExtractor());
    }
    
    /**
     * Calculate area of Weighted ROC curve based on the PerformanceObject List.
     * 
     * @param weightedRoc PerformanceObject List contains Weighted ROC curve data.
     * @return area under Weighted ROC.
     */
    public static double ofWeightedRoc(List<PerformanceObject> weightedRoc) {
        return calculateArea(weightedRoc, Performances.getWeightedFprExtractor(),Performances.getWeightedRecallExtractor());
    }
    
    /**
     * Calculate area of PR curve based on the PerformanceObject List.
     * 
     * @param pr PerformanceObject List contains PR curve data.
     * @return area under PR.
     */
    public static double ofPr(List<PerformanceObject> pr) {
        return calculateArea(pr, Performances.getRecallExtractor(), Performances.getPrecisionExtractor());
    }
    
    /**
     * Calculate area of Weighted PR curve based on the PerformanceObject List.
     * 
     * @param weightedPr PerformanceObject List contains Weighted PR curve data.
     * @return area under Weighted PR.
     */
    public static double ofWeightedPr(List<PerformanceObject> weightedPr) {
        return calculateArea(weightedPr, Performances.getWeightedRecallExtractor(), Performances.getWeightedPrecisionExtractor());
    }

    /**
     * Calculate curve area based on the PerformanceObject List and given extractor.
     * 
     * @param perform PerformanceObject List contains curve data.
     * @param xExtractor PerformanceExtractor instance used extract x of point PerformanceObject.
     * @param yExtractor PerformanceExtractor instance used extract y of point PerformanceObject.
     * @return area under curve.
     */
    public static double calculateArea(List<PerformanceObject> perform,
                                       PerformanceExtractor xExtractor, PerformanceExtractor yExtractor) {
        if(perform.size() < 2) {
            throw new IllegalArgumentException("We need at least 2 point to calculate area.");
        }
        
        // accumulate the trapezoid area of every successive two points in the curve.
        Iterator<PerformanceObject> iterator = perform.iterator();
        double sum = 0.0;
        PerformanceObject per = iterator.next();
        double x1 = xExtractor.extract(per);
        double y1 = yExtractor.extract(per);
        double x2;
        double y2;
        while(iterator.hasNext()) {
            per = iterator.next();
            x2 = xExtractor.extract(per);
            y2 = yExtractor.extract(per);
            sum += trapezoid(x1, y1, x2, y2);
            x1 = x2;
            y1 = y2;
        }
        
        return sum;
    }
    
}
