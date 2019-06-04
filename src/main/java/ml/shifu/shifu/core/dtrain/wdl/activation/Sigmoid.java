/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wdl.activation;

import ml.shifu.shifu.core.dtrain.AssertUtils;

/**
 * Typical sigmoid activation implementation.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Sigmoid extends Activation {

    /**
     * Last forward results which saved in forward but used in backward.
     */
    private double[] lastForward;

    @Override
    public double[] forward(double[] in) {
        AssertUtils.assertNotNull(in);

        double[] results = new double[in.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = (double) (1 / (1 + Math.min(1.0E19, Math.exp(-1 * in[i]))));
        }

        // temp saved for backward usage
        this.lastForward = results;
        return results;
    }

    @Override
    public double[] backward(double[] out) {
        AssertUtils.assertDoubleArrayNotNullAndLengthEqual(out, lastForward);

        double[] results = new double[out.length];
        for(int i = 0; i < results.length; i++) {
            results[i] = out[i] * lastForward[i] * (1f - lastForward[i]);
        }
        return results;
    }

}
