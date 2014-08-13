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
package ml.shifu.core.di.builtin;

import java.util.List;

/**
 * KV/IS calculator
 */
public class KSIVCalculator {

    private double ks;
    private double iv;

    public double getKS() {
        return ks;
    }

    public double getIV() {
        return iv;
    }

    public void calculateKSIV(List<Integer> h0, List<Integer> h1) {

        int numBins = h0.size();

        double sum0 = 0.0;
        double sum1 = 0.0;
        double cumP = 0.0;
        double cumQ = 0.0;
        double iv = 0.0;
        double ks = 0.0;

        for (int i = 0; i < numBins; i++) {
            sum0 += h0.get(i);
            sum1 += h1.get(i);
        }

        if (sum0 == 0 || sum1 == 0) {
            this.ks = -1;
            this.iv = -1;
            return;
        }

        for (int i = 0; i < numBins; i++) {
            double cnt0 = h0.get(i);
            double cnt1 = h1.get(i);
            //double rate = 0.0;
            //if (cnt0 > 0) {
            //	rate = cnt1 / cnt0;
            //}
            double p = cnt1 / sum1;
            double q = cnt0 / sum0;
            double EPS = 1e-10;
            double woe = Math.log((p + EPS) / (q + EPS));
            iv += (p - q) * woe;
            cumP += p;
            cumQ += q;
            double tmpKS = Math.abs(cumP - cumQ);
            if (ks < tmpKS) {
                ks = tmpKS;
            }
        }
        this.ks = ks * 100;
        this.iv = iv;
        //System.out.println(config.getName()+","+ks+","+iv);
    }

}
