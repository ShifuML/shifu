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
package ml.shifu.shifu.core;

import java.util.Arrays;

import org.apache.commons.lang.StringUtils;
import org.encog.mathutil.BoundMath;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class LR extends BasicML {

    /**
     * Serial id.
     */
    private static final long serialVersionUID = 1L;

    private double[] weights;

    public LR(double[] weights) {
        this.weights = weights;
    }

    public final MLData compute(final MLData input) {
        MLData result = new BasicMLData(1);
        double score = this.sigmoid(input.getData(), this.weights);
        result.setData(0, score);
        return result;
    }

    public int getInputCount() {
        return this.weights.length;
    }

    @Override
    public String toString() {
        return Arrays.toString(this.weights);
    }

    /**
     * Compute sigmoid value by dot operation of two vectors.
     */
    private double sigmoid(double[] inputs, double[] weights) {
        double value = 0.0d;
        for(int i = 0; i < weights.length; i++) {
            value += weights[i] * inputs[i];
        }

        return 1.0d / (1.0d + BoundMath.exp(-1 * value));
    }

    @Override
    public void updateProperties() {

    }

    public static LR loadFromString(String input) {
        String target = StringUtils.remove(StringUtils.remove(input, '['), ']');
        String[] ws = target.split(",");
        double[] weights = new double[ws.length];
        int index = 0;
        for(String weight: ws) {
            weights[index++] = Double.parseDouble(weight);
        }
        return new LR(weights);
    }

}
