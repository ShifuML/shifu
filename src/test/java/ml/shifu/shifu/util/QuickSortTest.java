/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.util;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.*;

/**
 * QuickSortTest class
 */
public class QuickSortTest {

    @Test
    public void testQuickSortComparable() {
        List<Integer> objList = null;
        QuickSort.sort(objList);

        List<Integer> dataList = new ArrayList<Integer>();

        QuickSort.sort(dataList);
        Assert.assertEquals("[]", dataList.toString());

        dataList.add(3);
        QuickSort.sort(dataList);
        Assert.assertEquals("[3]", dataList.toString());

        dataList.add(2);
        dataList.add(1);
        dataList.add(5);
        dataList.add(4);

        QuickSort.sort(dataList);
        Assert.assertEquals("[1, 2, 3, 4, 5]", dataList.toString());

        List<String> strList = new ArrayList<String>();
        strList.add("m");
        strList.add("z");
        strList.add("c");
        strList.add("g");
        strList.add("k");

        QuickSort.sort(strList);
        Assert.assertEquals("[c, g, k, m, z]", strList.toString());
    }

    @Test
    public void testQuickSortComparator() {
        QuickSort.sort(null, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                return left.compareTo(right);
            }
        });

        List<Integer> dataList = new ArrayList<Integer>();

        QuickSort.sort(dataList, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                return left.compareTo(right);
            }
        });
        Assert.assertEquals("[]", dataList.toString());

        dataList.add(3);
        QuickSort.sort(dataList, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                return left.compareTo(right);
            }
        });
        Assert.assertEquals("[3]", dataList.toString());

        dataList.add(2);
        dataList.add(1);
        dataList.add(5);
        dataList.add(4);

        QuickSort.sort(dataList, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                return left.compareTo(right);
            }
        });
        Assert.assertEquals("[1, 2, 3, 4, 5]", dataList.toString());

        List<String> strList = new ArrayList<String>();
        strList.add("m");
        strList.add("z");
        strList.add("c");
        strList.add("g");
        strList.add("k");

        QuickSort.sort(strList, new Comparator<String>() {
            @Override
            public int compare(String left, String right) {
                return left.compareTo(right);
            }
        });
        Assert.assertEquals("[c, g, k, m, z]", strList.toString());
    }

    @Test
    public void testPerformance() {
        final int SIZE = 100000;
        Random rd = new Random(System.currentTimeMillis());
        List<Integer> dataListA = new ArrayList<Integer>(SIZE);
        List<Integer> dataListB = new ArrayList<Integer>(SIZE);
        for (int i = 0; i < SIZE; i++) {
            int seed = rd.nextInt(SIZE);
            dataListA.add(seed);
            dataListB.add(seed);
        }

        long start = System.currentTimeMillis();
        Collections.sort(dataListA);
        long end = System.currentTimeMillis();
        long timeConsumptionA = (end - start);

        start = System.currentTimeMillis();
        QuickSort.sort(dataListB);
        end = System.currentTimeMillis();
        long timeConsumptionB = (end - start);

        System.out.println(timeConsumptionA + "  " + timeConsumptionB);
        // Sometimes Collections.sort may run faster than QuickSort
        Assert.assertTrue((double)timeConsumptionB/timeConsumptionA < 5.0d);
    }
}
