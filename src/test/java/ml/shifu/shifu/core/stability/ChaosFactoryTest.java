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


import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.junit.Assert;
import org.junit.Test;

public class ChaosFactoryTest {
    @Test
    public void testGetChaosType() {
        Environment.setProperty(Constants.CHAOS_TYPE, "null");
        Environment.setProperty(Constants.CHAOS_COLUMNS, "");
        Assert.assertEquals(ChaosFactory.getInstance().getChaosType().getName(), ChaosType.NULL_VALUE.getName());
    }
}