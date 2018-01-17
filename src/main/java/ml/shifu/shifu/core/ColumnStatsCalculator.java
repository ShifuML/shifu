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
package ml.shifu.shifu.core;

import java.util.ArrayList;
import java.util.List;

/**
 * To compute ks, iv and woe values.
 */
public final class ColumnStatsCalculator {

    private ColumnStatsCalculator() {
    }

    private final static double EPS = 1e-10;

    public static <T extends Number> ColumnMetrics calculateColumnMetrics(List<T> negative, List<T> positive) {
        assert negative != null && positive != null && negative.size() == positive.size();

        int numBins = negative.size();

        double sumN = 0.0;
        double sumP = 0.0;
        double cumN = 0.0;
        double cumP = 0.0;
        double iv = 0.0;
        double ks = 0.0;

        for(int i = 0; i < numBins; i++) {
            sumN += negative.get(i).doubleValue();
            sumP += positive.get(i).doubleValue();
        }

        if(sumN == 0 || sumP == 0) {
            return null;
        }

        double woe = Math.log((sumP + EPS) / (sumN + EPS));

        List<Double> binningWoe = new ArrayList<Double>(numBins);

        for(int i = 0; i < numBins; i++) {
            double cntN = negative.get(i).doubleValue();
            double cntP = positive.get(i).doubleValue();
            double p = cntP / sumP;
            double n = cntN / sumN;
            // TODO merge bin with p or q = 0 ???
            double woePerBin = Math.log((p + EPS) / (n + EPS));
            binningWoe.add(woePerBin);
            iv += (p - n) * woePerBin;
            cumP += p;
            cumN += n;
            double tmpKS = Math.abs(cumP - cumN);
            if(ks < tmpKS) {
                ks = tmpKS;
            }
        }

        return new ColumnMetrics(ks * 100, iv, woe, binningWoe);
    }

    public static ColumnMetrics calculateColumnMetrics(long[] negative, long[] positive) {
        assert negative != null && positive != null && negative.length == positive.length;

        int numBins = negative.length;

        double sumN = 0.0;
        double sumP = 0.0;
        double cumN = 0.0;
        double cumP = 0.0;
        double iv = 0.0;
        double ks = 0.0;

        for(int i = 0; i < numBins; i++) {
            sumN += negative[i];
            sumP += positive[i];
        }

        if(sumN == 0 || sumP == 0) {
            return null;
        }

        double woe = Math.log((sumN + EPS) / (sumP + EPS));

        List<Double> binningWoe = new ArrayList<Double>(numBins);

        for(int i = 0; i < numBins; i++) {
            double cntN = negative[i];
            double cntP = positive[i];
            double p = cntP / sumP;
            double n = cntN / sumN;
            // TODO merge bin with p or q = 0 ???
            double woePerBin = Math.log((n + EPS) / (p + EPS));
            binningWoe.add(woePerBin);
            iv += (n - p) * woePerBin;
            cumP += p;
            cumN += n;
            double tmpKS = Math.abs(cumP - cumN);
            if(ks < tmpKS) {
                ks = tmpKS;
            }
        }

        return new ColumnMetrics(ks * 100, iv, woe, binningWoe);
    }

    public static ColumnMetrics calculateColumnMetrics(double[] negative, double[] positive) {
        assert negative != null && positive != null && negative.length == positive.length;

        int numBins = negative.length;

        double sumN = 0.0;
        double sumP = 0.0;
        double cumN = 0.0;
        double cumP = 0.0;
        double iv = 0.0;
        double ks = 0.0;

        for(int i = 0; i < numBins; i++) {
            sumN += negative[i];
            sumP += positive[i];
        }

        if(sumN == 0 || sumP == 0) {
            return null;
        }

        double woe = Math.log((sumN + EPS) / (sumP + EPS));

        List<Double> binningWoe = new ArrayList<Double>(numBins);

        for(int i = 0; i < numBins; i++) {
            double cntN = negative[i];
            double cntP = positive[i];
            double p = cntP / sumP;
            double n = cntN / sumN;
            // TODO merge bin with p or q = 0 ???
            double woePerBin = Math.log((n + EPS) / (p + EPS));
            binningWoe.add(woePerBin);
            iv += (n - p) * woePerBin;
            cumP += p;
            cumN += n;
            double tmpKS = Math.abs(cumP - cumN);
            if(ks < tmpKS) {
                ks = tmpKS;
            }
        }

        return new ColumnMetrics(ks * 100, iv, woe, binningWoe);
    }

    /**
     * From link {@literal <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm">Kurtosis</a>}
     * 
     * @param count
     *            count
     * @param mean
     *            mean
     * @param stdDev
     *            stdDev
     * @param sum
     *            sum
     * @param squaredSum
     *            squaredSum
     * @param tripleSum
     *            tripleSum
     * @param quarticSum
     *            quarticSum
     * @return Kurtosis value
     */
    public static double computeKurtosis(long count, double mean, double stdDev, double sum, double squaredSum,
            double tripleSum, double quarticSum) {
        return (quarticSum - 4 * tripleSum * mean + 6 * squaredSum * mean * mean - 4 * sum * mean * mean * mean + count
                * mean * mean * mean * mean)
                / (count * stdDev * stdDev * stdDev * stdDev);
    }

    /**
     * From link {@literal <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm">Skewness</a>}
     * 
     * @param count
     *            count
     * @param mean
     *            mean
     * @param stdDev
     *            stdDev
     * @param sum
     *            sum
     * @param squaredSum
     *            squaredSum
     * @param tripleSum
     *            tripleSum
     * @return skewness value
     */
    public static double computeSkewness(long count, double mean, double stdDev, double sum, double squaredSum,
            double tripleSum) {
        return (tripleSum - 3 * squaredSum * mean + 3 * mean * mean * sum - count * mean * mean * mean)
                / (count * stdDev * stdDev * stdDev);
    }

    public static class ColumnMetrics {

        public ColumnMetrics(double ks, double iv, double woe, List<Double> binningWoe) {
            this.ks = ks;
            this.iv = iv;
            this.woe = woe;
            this.binningWoe = binningWoe;
        }

        private final double ks;

        private final double iv;

        private final double woe;

        private final List<Double> binningWoe;

        /**
         * @return the ks
         */
        public double getKs() {
            return ks;
        }

        /**
         * @return the iv
         */
        public double getIv() {
            return iv;
        }

        /**
         * @return the woe
         */
        public double getWoe() {
            return woe;
        }

        /**
         * @return the binningWoe
         */
        public List<Double> getBinningWoe() {
            return binningWoe;
        }
    }

}
