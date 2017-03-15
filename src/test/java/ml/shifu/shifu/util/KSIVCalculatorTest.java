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
package ml.shifu.shifu.util;

import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;

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

    @Test
    public void test() {
        List<Integer> a = Arrays.asList(new Integer[] { 1, 2, 3, 4, 5, 6 });
        List<Integer> b = Arrays.asList(new Integer[] { 2, 2, 5, 5, 5, 5 });

        ColumnMetrics columnMetrics = ColumnStatsCalculator.calculateColumnMetrics(a, b);

        Assert.assertEquals(df.format(columnMetrics.getIv()), "0.08");
        Assert.assertEquals(df.format(columnMetrics.getKs()), "10.71");

    }
}
