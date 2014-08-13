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
package ml.shifu.core.util;

import ml.shifu.core.di.builtin.KSIVCalculator;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;


/**
 * KSIVCalculatorTest class
 */
public class KSIVCalculatorTest {

    private static final DecimalFormat df = new DecimalFormat("0.00");
    private KSIVCalculator calc = new KSIVCalculator();

    @Test
    public void test() {
        List<Integer> a = Arrays.asList(1, 2, 3, 4, 5, 6);
        List<Integer> b = Arrays.asList(2, 2, 5, 5, 5, 5);

        calc.calculateKSIV(a, b);

        Assert.assertEquals(df.format(calc.getIV()), "0.08");
        Assert.assertEquals(df.format(calc.getKS()), "10.71");

    }
}
