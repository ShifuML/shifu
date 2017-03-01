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
package ml.shifu.shifu.container;

public class ConfusionMatrixObject {

    public ConfusionMatrixObject() {
        this.tp = 0.0;
        this.fn = 0.0;
        this.fp = 0.0;
        this.tn = 0.0;
        this.weightedFn = 0.0;
        this.weightedFp = 0.0;
        this.weightedTn = 0.0;
        this.weightedTp = 0.0;
    }

    public ConfusionMatrixObject(ConfusionMatrixObject cmo) {
        this.tp = cmo.tp;
        this.fn = cmo.fn;
        this.fp = cmo.fp;
        this.tn = cmo.tn;
        this.weightedFn = cmo.weightedFn;
        this.weightedFp = cmo.weightedFp;
        this.weightedTn = cmo.weightedTn;
        this.weightedTp = cmo.weightedTp;
    }

    private double tp, fp, tn, fn, weightedTp, weightedFp, weightedTn, weightedFn;

    private double score;

    public double getTp() {
        return tp;
    }

    public void setTp(double tp) {
        this.tp = tp;
    }

    public double getFp() {
        return fp;
    }

    public void setFp(double fp) {
        this.fp = fp;
    }

    public double getTn() {
        return tn;
    }

    public void setTn(double tn) {
        this.tn = tn;
    }

    public double getFn() {
        return fn;
    }

    public void setFn(double fn) {
        this.fn = fn;
    }

    public double getWeightedTp() {
        return weightedTp;
    }

    public void setWeightedTp(double weightedTp) {
        this.weightedTp = weightedTp;
    }

    public double getWeightedFp() {
        return weightedFp;
    }

    public void setWeightedFp(double weightedFp) {
        this.weightedFp = weightedFp;
    }

    public double getWeightedTn() {
        return weightedTn;
    }

    public void setWeightedTn(double weightedTn) {
        this.weightedTn = weightedTn;
    }

    public double getWeightedFn() {
        return weightedFn;
    }

    public void setWeightedFn(double weightedFn) {
        this.weightedFn = weightedFn;
    }

    public double getTotal() {
        return this.tp + this.tn + this.fn + this.fp;
    }

    public double getWeightedTotal() {
        return this.weightedTp + this.weightedTn + this.weightedFn + this.weightedFp;
    }

    public double getPosTotal() {
        return this.getTp() + this.getFn();
    }

    public double getWeightPosTotal() {
        return this.getWeightedTp() + this.getWeightedFn();
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

}
