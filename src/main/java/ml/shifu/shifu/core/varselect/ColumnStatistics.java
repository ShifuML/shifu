/*
 * Copyright [2012-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.varselect;

/**
 * Sensitivity analysis report format, including mean, rms and variance.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class ColumnStatistics {

    private double mean;

    private double rms;

    private double variance;

    public ColumnStatistics() {
    }

    public ColumnStatistics(double mean, double rms, double variance) {
        this.mean = mean;
        this.rms = rms;
        this.variance = variance;
    }

    /**
     * @return the mean
     */
    public double getMean() {
        return mean;
    }

    /**
     * @param mean
     *            the mean to set
     */
    public void setMean(double mean) {
        this.mean = mean;
    }

    /**
     * @return the rms
     */
    public double getRms() {
        return rms;
    }

    /**
     * @param rms
     *            the rms to set
     */
    public void setRms(double rms) {
        this.rms = rms;
    }

    /**
     * @return the variance
     */
    public double getVariance() {
        return variance;
    }

    /**
     * @param variance
     *            the variance to set
     */
    public void setVariance(double variance) {
        this.variance = variance;
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return "ColumnStatistics [mean=" + mean + ", rms=" + rms + ", variance=" + variance + "]";
    }

}
