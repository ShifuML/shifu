/**
 * Copyright [2012-2013] eBay Software Foundation
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

import java.util.ArrayList;
import java.util.List;

/**
 * WOE calculator
 */
public class WOECalculator {

    private static final double EPS = 1e-10;

    public static List<Double> calculate(Object[] pos, Object[] neg) {
        Double[] tmpPos = new Double[pos.length];
        Double[] tmpNeg = new Double[neg.length];

        for (int i = 0; i < pos.length; i++) {
            tmpPos[i] = Double.valueOf(pos[i].toString());
            tmpNeg[i] = Double.valueOf(neg[i].toString());
        }

        return calculate(tmpPos, tmpNeg);
    }

    public static List<Double> calculate(Integer[] pos, Integer[] neg) {
        Double[] tmpPos = new Double[pos.length];
        Double[] tmpNeg = new Double[neg.length];

        for (int i = 0; i < pos.length; i++) {
            tmpPos[i] = Double.valueOf(pos[i]);
            tmpNeg[i] = Double.valueOf(neg[i]);
        }

        return calculate(tmpPos, tmpNeg);
    }

    public static List<Double> calculate(Double[] pos, Double[] neg) {

        List<Double> woe = new ArrayList<Double>();

        int posSize = pos.length, negSize = neg.length;

        if (posSize != negSize) {
            throw new RuntimeException("Inconsistent Length: Positive=" + posSize + ", Negative=" + negSize);
        }

        double sumPos = 0.0;
        double sumNeg = 0.0;

        for (int i = 0; i < posSize; i++) {
            sumPos += pos[i];
            sumNeg += neg[i];
        }


        for (int i = 0; i < posSize; i++) {
            woe.add(Math.log((pos[i] / sumPos + EPS) / (neg[i] / sumNeg + EPS)));
        }

        return woe;
    }

}
