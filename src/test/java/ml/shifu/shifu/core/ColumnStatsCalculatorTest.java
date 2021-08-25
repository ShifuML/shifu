/*
 * Copyright [2013-2015] PayPal Software Foundation
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.testng.annotations.Test;

/**
 * Created by zhanhu on 11/10/16.
 */
public class ColumnStatsCalculatorTest {

    @Test
    public void testIv() throws IOException {
        // List<Integer> negativeList = Arrays.asList(new Integer[]{75889, 6799980});
        // List<Integer> positiveList = Arrays.asList(new Integer[]{6334, 2328});

        long[] negativeList = new long[]{23017, 17781, 394493, 1820837, 1378687, 997797, 777811, 1458379, 2688, 4379};
        long[] positiveList = new long[]{874, 818, 881, 823, 883, 882, 876, 1044, 717, 864};

        ColumnStatsCalculator.ColumnMetrics status = ColumnStatsCalculator.calculateColumnMetrics(negativeList, positiveList);
        System.out.println(status.getIv());
        System.out.println(status.getKs());
        System.out.println(status.getBinningWoe());

        double[] negativeDoubleList = new double[negativeList.length];
        for (int i = 0; i < negativeDoubleList.length; i ++) {
            negativeDoubleList[i] = negativeList[i];
        }
        double[] positiveDoubleList = new double[negativeList.length];
        for (int i = 0; i < positiveDoubleList.length; i ++) {
            positiveDoubleList[i] = positiveList[i];
        }
        status = ColumnStatsCalculator.calculateColumnMetrics(negativeDoubleList, positiveDoubleList);
        System.out.println(status.getIv());
        System.out.println(status.getKs());
        System.out.println(status.getBinningWoe());

        List<Long> negativeLongList = new ArrayList<>();
        for (long v : negativeList) {
            negativeLongList.add(v);
        }
        List<Long> positiveLongList = new ArrayList<>();
        for (long v : positiveList) {
            positiveLongList.add(v);
        }
        status = ColumnStatsCalculator.calculateColumnMetrics(negativeLongList, positiveLongList);
        System.out.println(status.getIv());
        System.out.println(status.getKs());
        System.out.println(status.getBinningWoe());
    }

}
