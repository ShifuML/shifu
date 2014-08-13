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
package ml.shifu.core.container;

import java.util.Comparator;

/**
 * Model score result object
 */
public class ModelResultObject {

    private double score;
    private String tag;
    private double weight = 1.0;


    public ModelResultObject(double score, String tag, double weight) {
        this.score = score;
        this.tag = tag;
        this.weight = weight;
    }

    public Double getWeight() {
        return weight;
    }

    public double getScore() {
        return score;
    }

    public String getTag() {
        return tag;
    }

    // Others
    public String toString() {
        return "(" + this.score + ", " + this.tag + ", " + this.weight + ")";
    }

    // Comparator, descends
    public static class ModelResultObjectComparator implements Comparator<ModelResultObject> {
        public int compare(ModelResultObject a, ModelResultObject b) {
            return Double.compare(b.score, a.score);
        }
    }

}
