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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.ValueObject;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.core.Binning.BinningDataType;
import org.apache.commons.collections.map.HashedMap;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class BinningTest {

    private Binning binN, binC, binA;

    private List<ValueObject> voList;

    private Random rdm;

    private int numBin = 10;

    @BeforeClass
    public void setUp() {
        voList = new ArrayList<ValueObject>();
        rdm = new Random(new Date().getTime());

    }

    @Test
    public void numericalTest() {

        for (int i = 0; i < 5000; i++) {
            ValueObject vo = new ValueObject();
            vo.setValue(rdm.nextDouble());
            vo.setRaw(Integer.toString(rdm.nextInt(100)));
            vo.setTag(Integer.toString(rdm.nextInt(2)));
            vo.setWeight(rdm.nextDouble());
            voList.add(vo);
        }

        List<String> posTags = new ArrayList<String>();
        posTags.add("1");
        List<String> negTag = new ArrayList<String>();
        negTag.add("0");

        binN = new Binning(posTags, negTag, BinningDataType.Numerical, voList);

        binN.setMaxNumOfBins(numBin);
        binN.setBinningMethod(BinningMethod.EqualPositive);
        binN.setAutoTypeThreshold(3);
        binN.setMergeEnabled(true);
        binN.doBinning();

        binN.setBinningMethod(BinningMethod.EqualTotal);
        binN.doBinning();

        binN.setBinningMethod(BinningMethod.EqualInterval);
        binN.doBinning();

        Assert.assertEquals(binN.getNumBins(), numBin);
    }

    @Test
    public void categoricalTest() {

        Set<String> categorySet = new HashSet<String>();

        for (int i = 0; i < 1000; i++) {
            ValueObject vo = new ValueObject();
            vo.setValue(rdm.nextDouble());
            String input = Integer.toString(rdm.nextInt(100));
            categorySet.add(input);
            vo.setRaw(input);
            vo.setTag(Integer.toString(rdm.nextInt(2)));
            vo.setWeight(rdm.nextDouble());
            voList.add(vo);
        }

        List<String> posTags = new ArrayList<String>();
        posTags.add("1");
        List<String> negTag = new ArrayList<String>();
        negTag.add("0");
        binC = new Binning(posTags, negTag, BinningDataType.Categorical, voList);

        binC.setMaxNumOfBins(6);
        binC.setBinningMethod(BinningMethod.EqualPositive);
        binC.setAutoTypeThreshold(3);
        binC.setMergeEnabled(true);
        binC.doBinning();

        binC.setBinningMethod(BinningMethod.EqualTotal);
        binC.doBinning();

        binC.setBinningMethod(BinningMethod.EqualInterval);
        binC.doBinning();

        //TODO 
        //Assert.assertEquals(categorySet.size(), binC.getNumBins());
    }

    @Test
    public void autoTest() {

        Set<String> categorySet = new HashSet<String>();

        for (int i = 0; i < 3; i++) {
            ValueObject vo = new ValueObject();
            //vo.setValue(rdm.nextDouble());
            String input = Integer.toString(rdm.nextInt(100));
            categorySet.add(input);
            vo.setRaw(input);
            vo.setTag(Integer.toString(rdm.nextInt(2)));
            vo.setWeight(rdm.nextDouble());
            voList.add(vo);
        }

        List<String> posTags = new ArrayList<String>();
        posTags.add("1");
        List<String> negTag = new ArrayList<String>();
        negTag.add("0");
        binA = new Binning(posTags, negTag, BinningDataType.Auto, voList);

        binA.setMaxNumOfBins(6);
        binA.setBinningMethod(BinningMethod.EqualPositive);
        binA.setAutoTypeThreshold(1002);
        binA.setMergeEnabled(true);
        binA.doBinning();

        binA.setBinningMethod(BinningMethod.EqualTotal);
        binA.doBinning();

        binA.setBinningMethod(BinningMethod.EqualInterval);
        binA.doBinning();

        //TODO test case
    }

    @Test
    public void testSortMap() {
        Map<String, Double> categoryFraudRateMap = new HashedMap();
        categoryFraudRateMap.put("a", 10.0);
        categoryFraudRateMap.put("b", 10.0);
        categoryFraudRateMap.put("c", 10.0);

        MapComparator cmp = new MapComparator(categoryFraudRateMap);
        Map<String, Double> sortedCategoryFraudRateMap = new TreeMap<String, Double>(cmp);
        sortedCategoryFraudRateMap.putAll(categoryFraudRateMap);
        Assert.assertEquals(sortedCategoryFraudRateMap.keySet().size(), 1);

        sortedCategoryFraudRateMap = categoryFraudRateMap.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(
                Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                        (e1, e2) -> e1, LinkedHashMap::new));
        Assert.assertEquals(sortedCategoryFraudRateMap.keySet().size(), 3);
    }

    /**
     * comparator for map
     */
    public static class MapComparator implements Comparator<String>, Serializable {

        private static final long serialVersionUID = 2178035954425107063L;

        Map<String, Double> base;

        public MapComparator(Map<String, Double> base) {
            this.base = base;
        }

        public int compare(String a, String b) {
            return base.get(a).compareTo(base.get(b));
        }
    }
}
