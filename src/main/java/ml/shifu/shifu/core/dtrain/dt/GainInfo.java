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
package ml.shifu.shifu.core.dtrain.dt;

import java.util.List;

/**
 * Gain information bean.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class GainInfo {

    private double gain;

    private double impurity;

    private Predict predict;

    private double leftImpurity;

    private double rightImpurity;

    private Predict leftPredict;

    private Predict rightPredict;

    private double wgtCnt;

    private Split split;

    public GainInfo(double gain, double impurity, Predict predict, double leftImpurity, double rightImpurity,
            Predict leftPredict, Predict rightPredict, Split split, double wgtCnt) {
        this.gain = gain;
        this.impurity = impurity;
        this.predict = predict;
        this.leftImpurity = leftImpurity;
        this.rightImpurity = rightImpurity;
        this.leftPredict = leftPredict;
        this.rightPredict = rightPredict;
        this.split = split;
        this.wgtCnt = wgtCnt;
    }

    /**
     * @return the gain
     */
    public double getGain() {
        return gain;
    }

    /**
     * @return the impurity
     */
    public double getImpurity() {
        return impurity;
    }

    /**
     * @return the predict
     */
    public Predict getPredict() {
        return predict;
    }

    /**
     * @return the leftImpurity
     */
    public double getLeftImpurity() {
        return leftImpurity;
    }

    /**
     * @return the rightImpurity
     */
    public double getRightImpurity() {
        return rightImpurity;
    }

    /**
     * @return the leftPredict
     */
    public Predict getLeftPredict() {
        return leftPredict;
    }

    /**
     * @return the rightPredict
     */
    public Predict getRightPredict() {
        return rightPredict;
    }

    /**
     * @return the split
     */
    public Split getSplit() {
        return split;
    }

    /**
     * @param gain
     *            the gain to set
     */
    public void setGain(double gain) {
        this.gain = gain;
    }

    /**
     * @param impurity
     *            the impurity to set
     */
    public void setImpurity(double impurity) {
        this.impurity = impurity;
    }

    /**
     * @param predict
     *            the predict to set
     */
    public void setPredict(Predict predict) {
        this.predict = predict;
    }

    /**
     * @param leftImpurity
     *            the leftImpurity to set
     */
    public void setLeftImpurity(double leftImpurity) {
        this.leftImpurity = leftImpurity;
    }

    /**
     * @param rightImpurity
     *            the rightImpurity to set
     */
    public void setRightImpurity(double rightImpurity) {
        this.rightImpurity = rightImpurity;
    }

    /**
     * @param leftPredict
     *            the leftPredict to set
     */
    public void setLeftPredict(Predict leftPredict) {
        this.leftPredict = leftPredict;
    }

    /**
     * @param rightPredict
     *            the rightPredict to set
     */
    public void setRightPredict(Predict rightPredict) {
        this.rightPredict = rightPredict;
    }

    /**
     * @param split
     *            the split to set
     */
    public void setSplit(Split split) {
        this.split = split;
    }

    /**
     * @return the wgtCnt
     */
    public double getWgtCnt() {
        return wgtCnt;
    }

    /**
     * @param wgtCnt
     *            the wgtCnt to set
     */
    public void setWgtCnt(double wgtCnt) {
        this.wgtCnt = wgtCnt;
    }

    /**
     * Return {@link GainInfo} instance with max gain value.
     * 
     * @param gainList
     *            the gain info list
     * @return max {@link GainInfo} instance
     */
    public static GainInfo getGainInfoByMaxGain(List<GainInfo> gainList) {
        double maxGain = Double.MIN_VALUE;
        int maxGainIndex = -1;
        for(int i = 0; i < gainList.size(); i++) {
            double gain = gainList.get(i).getGain();
            if(gain > maxGain) {
                maxGain = gain;
                maxGainIndex = i;
            }
        }
        if(maxGainIndex == -1) {
            return null;
        }
        return gainList.get(maxGainIndex);
    }

    @Override
    public String toString() {
        return "GainInfo [gain=" + gain + ", impurity=" + impurity + ", predict=" + predict + ", leftImpurity="
                + leftImpurity + ", rightImpurity=" + rightImpurity + ", leftPredict=" + leftPredict
                + ", rightPredict=" + rightPredict + ", split=" + split + "]";
    }

}
