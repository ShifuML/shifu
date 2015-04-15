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
    
    private static Double trapezoid() {
        // TODO: calculate trapezoid area.
        return Double.valueOf(0);
    }
    
    public static Double calculate(List<PerformanceObject> curves) {
        Double area = Double.valueOf(0);
        // TODO: calculate area.
        return area;
    }
}
