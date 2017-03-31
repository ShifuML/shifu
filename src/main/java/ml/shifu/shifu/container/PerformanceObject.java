/*
 * Copyright [2012-2014] PayPal Software Foundation
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

/**
 * performance object
 */
public class PerformanceObject {

    public PerformanceObject() {
        this.binNum = 0;
        this.binLowestScore = 0.0;
        this.actionRate = 0.0;
        this.weightedActionRate = 0.0;

    }

    /**
     * bin number
     */
    public int binNum;

    /**
     * score
     */
    public double binLowestScore;

    /**
     * action to be execution
     */
    public double actionRate;

    /**
     * action with weights
     */
    public double weightedActionRate;

    /**
     * catch rate = Recall = tp / (tp+fn)
     */
    public double recall;

    /**
     * weighted recall rate
     */
    public double weightedRecall;

    /**
     * hit rate = Precision = tp /(tp+fp)
     */
    public double precision;

    /**
     * weight hit rate
     */
    public double weightedPrecision;

    /**
     * fall-out or false positive rate (FPR) = fp /(fp + tn)
     */
    public double fpr;

    public double weightedFpr;

    public double liftUnit;

    public double weightLiftUnit;

    public double scoreCount;

    public double scoreWgtCount;

    public double tp;
    public double fp;
    public double tn;
    public double fn;

    public double weightedTp;
    public double weightedFp;
    public double weightedTn;
    public double weightedFn;

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "PerformanceObject [binNum=" + binNum + ", binLowestScore=" + binLowestScore + ", actionRate="
                + actionRate + ", weightedActionRate=" + weightedActionRate + ", recall=" + recall
                + ", weightedRecall=" + weightedRecall + ", precision=" + precision + ", weightedPrecision="
                + weightedPrecision + ", fpr=" + fpr + ", weightedFpr=" + weightedFpr + ", liftUnit=" + liftUnit
                + ", weightLiftUnit=" + weightLiftUnit + ", scoreCount=" + scoreCount + ", scoreWgtCount="
                + scoreWgtCount + ", tp=" + tp + ", fp=" + fp + ", tn=" + tn + ", fn=" + fn + ", weightedTp="
                + weightedTp + ", weightedFp=" + weightedFp + ", weightedTn=" + weightedTn + ", weightedFn="
                + weightedFn + "]";
    }

}
