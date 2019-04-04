package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.core.dtrain.nn.ActivationPTANH;
import org.encog.engine.network.activation.ActivationTANH;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ActivationPTANHTest {

    @Test
    public void test() {
        ActivationTANH tanh = new ActivationTANH();
        ActivationPTANH ptanh = new ActivationPTANH();

        double[] inputs = new double[]{0.0d, 1.0d, -1.0d};
        ptanh.activationFunction(inputs, 0, 3);

        Assert.assertTrue(Math.abs(inputs[0] - 0.0) < 1e-6);
        Assert.assertTrue(Math.abs(inputs[1] - 0.7615941559557649d) < 1e-6);
        Assert.assertTrue(Math.abs(inputs[2] + 0.1903985389889412d) < 1e-6);

        double d = ptanh.derivativeFunction(0.0d, inputs[0]);
        Assert.assertTrue(Math.abs(d  - 0.25d) < 1e-6);
        d = ptanh.derivativeFunction(1.0d, inputs[1]);
        Assert.assertTrue(Math.abs(d  - tanh.derivativeFunction(1.0d, inputs[1])) < 1e-6);

        double[] t = new double[]{-1.0d};
        tanh.activationFunction(t, 0, 1);
        d = ptanh.derivativeFunction(-1.0d, inputs[2]);
        Assert.assertTrue(Math.abs(d  - 0.25 * tanh.derivativeFunction(-1.0d, t[0])) < 1e-6);
    }
}
