/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.eval;

import java.util.Iterator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Class for computing area under curve.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 */
public final class AreaUnderCurve {

    private AreaUnderCurve() {
    }

    private static final Logger LOG = LoggerFactory.getLogger(AreaUnderCurve.class);

    /**
     * Compute the area under the line connecting the two input points by the trapezoidal rule. The point
     * is stored as two double value which refer to x-coordinate and y-coordinate respectively.
     * 
     * <p>
     * Note: x2 is considered to be no less than x1, so that (x2 - x1) &gt;= 0 and the return value is always a nonnegative
     * </p>
     * 
     * @param x1
     *            x-coordinate of first point.
     * @param y1
     *            y-coordinate of first point.
     * @param x2
     *            x-coordinate of second point.
     * @param y2
     *            y-coordinate of second point.
     * @return trapezoid area.
     */
    public static double trapezoid(double x1, double y1, double x2, double y2) {
        return (y2 + y1) * (x2 - x1) / 2.0;
    }

    /**
     * Calculate area under ROC curve based on the PerformanceObject List.
     * 
     * @param roc
     *            PerformanceObject List contains ROC curve data.
     * @return area under ROC. Return 0 if input list is null or the size of list is less than 2.
     */
    public static double ofRoc(List<PerformanceObject> roc) {
        return calculateArea(roc, Performances.fpr(), Performances.recall());
    }

    /**
     * Calculate area under Weighted ROC curve based on the PerformanceObject List.
     * 
     * @param weightedRoc
     *            PerformanceObject List contains Weighted ROC curve data.
     * @return area under Weighted ROC. Return 0 if input list is null or the size of list is less than 2.
     */
    public static double ofWeightedRoc(List<PerformanceObject> weightedRoc) {
        return calculateArea(weightedRoc, Performances.weightedFpr(), Performances.weightedRecall());
    }

    /**
     * Calculate area under PR curve based on the PerformanceObject List.
     * 
     * @param pr
     *            PerformanceObject List contains PR curve data.
     * @return area under PR. Return 0 if input list is null or the size of list is less than 2.
     */
    public static double ofPr(List<PerformanceObject> pr) {
        return calculateArea(pr, Performances.recall(), Performances.precision());
    }

    /**
     * Calculate area under Weighted PR curve based on the PerformanceObject List.
     * 
     * @param weightedPr
     *            PerformanceObject List contains Weighted PR curve data.
     * @return area under Weighted PR. Return 0 if input list is null or the size of list is less than 2.
     */
    public static double ofWeightedPr(List<PerformanceObject> weightedPr) {
        return calculateArea(weightedPr, Performances.weightedRecall(), Performances.weightedPrecision());
    }

    /**
     * Calculate curve area by trapezoidal rule based on the given PerformanceObject List and extractor.
     * 
     * @param perform
     *            PerformanceObject List contains curve data.
     * @param xExtractor
     *            PerformanceExtractor instance used extract x of the point from PerformanceObject.
     * @param yExtractor
     *            PerformanceExtractor instance used extract y of the point from PerformanceObject.
     * @return the area under the curve. Return 0 if input list is null or the size of list is less than 2.
     * @throws IllegalArgumentException
     *             if the input xExtractor or yExtractor is null.
     */
    public static double calculateArea(List<PerformanceObject> perform, PerformanceExtractor xExtractor,
            PerformanceExtractor yExtractor) {
        if(perform == null) {
            LOG.warn("Input PerformanceObject List is null! Maybe you should check the input.");
            return 0;
        }

        if(perform.size() < 2) {
            LOG.warn("We need at least 2 point to calculate area! Maybe you should check the input.");
            return 0;
        }

        if(xExtractor == null || yExtractor == null) {
            throw new IllegalArgumentException("The xExtractor and yExtractor can't be null!");
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
