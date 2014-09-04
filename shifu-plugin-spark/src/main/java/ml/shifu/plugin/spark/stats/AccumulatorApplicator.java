/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.plugin.spark.stats;

import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;

import org.apache.spark.Accumulable;
import org.apache.spark.api.java.function.VoidFunction;

/**
 * The transform which takes a Accumulable object and only accumulates over an RDD
 * using the object.
 */
public class AccumulatorApplicator implements VoidFunction<String> {

    Accumulable<ColumnStateArray, String> accum;
    
    AccumulatorApplicator(Accumulable<ColumnStateArray, String> accum) {
        this.accum= accum;
    }
    
    public void call(String line) throws Exception {
        accum.add(line);
    }

}
