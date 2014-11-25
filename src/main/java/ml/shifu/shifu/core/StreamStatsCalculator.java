/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * StreamStatsCalculator class
 * 
 * @author zhanhu
 * @Oct 28, 2014
 *
 */
public class StreamStatsCalculator {
    
    private static final Logger log = LoggerFactory.getLogger(StreamStatsCalculator.class);

    private double sum = 0.0;
    private double squaredSum = 0.0;
    private double min = Double.MAX_VALUE;
    private double max = -Double.MAX_VALUE;
    private double mean = Double.NaN;
    private double stdDev = Double.NaN;
    private double median = Double.NaN;

    private Double threshold = 1e6;
    private Double EPS = 1e-6;

    private int validElementCnt = 0;
    
    
    
    /**
     * Constructor
     *
     * @param threshold
     */
    public StreamStatsCalculator(Double threshold) {
        this.threshold = threshold;
    }

    public void addData(double data) {
        if ( Double.isInfinite(data) || Double.isNaN(data) || Math.abs(data) > threshold) {
            log.warn("Invalid value - " + data);
            return;
        }
        
        validElementCnt ++;
        
        max = Math.max(max, data);
        min = Math.min(min, data);

        sum += data;
        squaredSum += data * data;
    }
    
    public void addData(double data, int frequency) {
        if ( frequency <= 0 ) {
            log.warn("Invalid frequency - " + frequency);
            return;
        }
        
        if (Double.isInfinite(data) || Double.isNaN(data) || Math.abs(data) > threshold) {
            log.warn("Invalid value - " + data);
            return;
        }
        
        validElementCnt += frequency;
        
        max = Math.max(max, data);
        min = Math.min(min, data);

        sum += data * frequency;
        squaredSum += data * data * frequency;
    }
    
    
    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }

    public double getMean() {
        if ( validElementCnt > 0 ) {
            mean = sum / validElementCnt;
        }
        
        return mean;
    }

    public double getStdDev() {
        if ( validElementCnt <= 1 || Double.isInfinite(sum) || Double.isInfinite(squaredSum) ) {
            return Double.NaN;
        }
        
        stdDev = Math.sqrt((squaredSum - (sum * sum) / validElementCnt + EPS) / ( validElementCnt  - 1));
        return stdDev;
    }

    public double getMedian() {
        return median;
    }
    
}