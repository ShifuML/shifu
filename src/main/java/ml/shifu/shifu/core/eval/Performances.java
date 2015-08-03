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

import ml.shifu.shifu.container.PerformanceObject;

/**
 * Factory class for getting some useful PerformanceExtractor implementation,
 * includes Fpr, WeightedFpr, Recall, WeightedRecall, Precision, WeightedPrecision.
 * 
 * @see Fpr
 * @see WeightedFpr
 * @see Recall
 * @see WeightedRecall
 * @see Precision
 * @see WeightedPrecision
 *
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 *
 */
public final class Performances {
    
    private Performances() {}
    
    private static Fpr fpr = new Fpr();
    private static WeightedFpr wFpr = new WeightedFpr();
    private static Recall recall = new Recall();
    private static WeightedRecall wRecall = new WeightedRecall();
    private static Precision precision = new Precision();
    private static WeightedPrecision wPrecision = new WeightedPrecision();

    public static PerformanceExtractor fpr() {
        return fpr;
    }
    
    public static PerformanceExtractor weightedFpr() {
        return wFpr;
    }
    
    public static PerformanceExtractor recall() {
        return recall;
    }
    
    public static PerformanceExtractor weightedRecall() {
        return wRecall;
    }

    public static PerformanceExtractor precision() {
        return precision;
    }

    public static PerformanceExtractor weightedPrecision() {
        return wPrecision;
    }
}

/**
 * Extractor used to extract fpr from PerformanceObject.
 */
class Fpr implements PerformanceExtractor {

    @Override
    public double extract(PerformanceObject perform) {
        return perform.fpr;
    }
    
}

/**
 * Extractor used to extract weightedFpr from PerformanceObject.
 */
class WeightedFpr implements PerformanceExtractor {
    
    @Override
    public double extract(PerformanceObject perform) {
        return perform.weightedFpr;
    }
    
}

/**
 * Extractor used to extract recall from PerformanceObject.
 */
class Recall implements PerformanceExtractor {
    
    @Override
    public double extract(PerformanceObject perform) {
        return perform.recall;
    }
    
}

/**
 * Extractor used to extract weightedRecall from PerformanceObject.
 */
class WeightedRecall implements PerformanceExtractor {
    
    @Override
    public double extract(PerformanceObject perform) {
        return perform.weightedRecall;
    }
    
}

/**
 * Extractor used to extract precision from PerformanceExtractor object.
 */
class Precision implements PerformanceExtractor {

    @Override
    public double extract(PerformanceObject perform) {
        return perform.precision;
    }

}

/**
 * Extractor used to extract weightedPrecision from PerformanceExtractor object.
 */
class WeightedPrecision implements PerformanceExtractor {

    @Override
    public double extract(PerformanceObject perform) {
        return perform.weightedPrecision;
    }

}
