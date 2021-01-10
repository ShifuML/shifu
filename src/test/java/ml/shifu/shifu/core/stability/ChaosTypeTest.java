/*
 * Copyright [2013-2021] PayPal Software Foundation
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
package ml.shifu.shifu.core.stability;

import ml.shifu.shifu.core.stability.algorithm.DeviationChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.NullValueChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.RandomChaosAlgorithm;
import org.junit.Assert;
import org.junit.Test;

public class ChaosTypeTest {
    @Test
    public void testGetChaosAlgorithm() {
        Assert.assertTrue(ChaosType.NULL_VALUE.getChaosAlgorithm() instanceof NullValueChaosAlgorithm);
        Assert.assertTrue(ChaosType.RANDOM_VALUE.getChaosAlgorithm() instanceof RandomChaosAlgorithm);
        Assert.assertTrue(ChaosType.DEVIATION_VALUE.getChaosAlgorithm() instanceof DeviationChaosAlgorithm);

    }

    @Test
    public void testFromName() {
        Assert.assertEquals(ChaosType.fromName("null").getName(), "null");
        Assert.assertNull(ChaosType.fromName("not_exist"));
    }
}