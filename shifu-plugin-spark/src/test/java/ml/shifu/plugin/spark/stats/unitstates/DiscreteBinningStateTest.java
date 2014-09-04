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
package ml.shifu.plugin.spark.stats.unitstates;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import junit.framework.Assert;
import ml.shifu.plugin.spark.stats.SerializedCategoricalValueObject;

import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.Extension;
import org.dmg.pmml.UnivariateStats;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class DiscreteBinningStateTest {
    DiscreteBinningUnitState state= new DiscreteBinningUnitState();
    
    @BeforeClass
    public void populateStatetest() {
        state.addData(new SerializedCategoricalValueObject("a", (double) 1, true));
        state.addData(new SerializedCategoricalValueObject("a", (double) 2, false));
        state.addData(new SerializedCategoricalValueObject("b", (double) 2, true));
        state.addData(new SerializedCategoricalValueObject("b", (double) -1, false));
        state.addData(new SerializedCategoricalValueObject("b", (double) 0, false));
        state.addData(new SerializedCategoricalValueObject("b", (double) 2, true));
    }
    
    // TODO: Null/ ClassCast tests
    
    @Test
    public void testCategoryHistNeg() {
        Map<String, Integer> m= state.getCategoryHistNeg();
        Assert.assertEquals(m.get("a"), (Integer)1);
        Assert.assertEquals(m.get("b"), (Integer)2);
    }

    @Test
    public void testCategoryHistPos() {
        Map<String, Integer> m= state.getCategoryHistPos();
        Assert.assertEquals(m.get("a"), (Integer)1);
        Assert.assertEquals(m.get("b"), (Integer)2);
    }

    @Test
    public  void testCategoryWeightPos() {
        Map<String, Double> m= state.getCategoryWeightPos();
        Assert.assertEquals(m.get("a"), 1.0);
        Assert.assertEquals(m.get("b"), 4.0);
        
    }

    @Test
    public  void testCategoryWeightNeg() {
        Map<String, Double> m= state.getCategoryWeightNeg();
        Assert.assertEquals(m.get("a"), 2.0);
        Assert.assertEquals(m.get("b"), -1.0);
    }
    
    @Test
    public void testCategorySet() {
        Set<String> s= state.getCategorySet();
        Assert.assertEquals(s.size(), 2);
        Assert.assertTrue(s.contains("a"));
        Assert.assertTrue(s.contains("b"));
    }

    
    @Test
    public void testPMML() {
        UnivariateStats us= new UnivariateStats();
        state.populateUnivariateStats(us, null);
        DiscrStats ds= us.getDiscrStats();
        List<Extension>extList= ds.getExtensions();
        // assert length
        Assert.assertEquals(extList.size(), 5);
        
        List<String> names= new ArrayList<String>();
        List<String> values= new ArrayList<String>();
        
        for(Extension ext: extList) {
            names.add(ext.getName());
            values.add(ext.getValue());
        }
        
        // TODO: parse values into doubles and compare
        
        List<Integer> binCountPos1= new ArrayList<Integer>();
        binCountPos1.add(1);
        binCountPos1.add(2);
        List<Integer> binCountPos2= new ArrayList<Integer>();
        binCountPos2.add(2);
        binCountPos2.add(1);
        
        List<Integer> binCountNeg1= new ArrayList<Integer>();
        binCountNeg1.add(1);
        binCountNeg1.add(2);
        List<Integer> binCountNeg2= new ArrayList<Integer>();
        binCountNeg2.add(2);
        binCountNeg2.add(1);
        
        List<Double> binWeightedCountPos1= new ArrayList<Double>();
        binWeightedCountPos1.add(1.0);
        binWeightedCountPos1.add(4.0);
        List<Double> binWeightedCountPos2= new ArrayList<Double>();
        binWeightedCountPos2.add(4.0);
        binWeightedCountPos2.add(1.0);

        List<Double> binWeightedCountNeg1= new ArrayList<Double>();
        binWeightedCountNeg1.add(-1.0);
        binWeightedCountNeg1.add(2.0);
        List<Double> binWeightedCountNeg2= new ArrayList<Double>();
        binWeightedCountNeg2.add(2.0);
        binWeightedCountNeg2.add(-1.0);

        List<Double> binPosRate1= new ArrayList<Double>();
        binPosRate1.add(0.5);
        binPosRate1.add(0.5);

        
        Assert.assertTrue(isEqualInteger(values, names, "BinCountPos", binCountPos1, binCountPos2));
        Assert.assertTrue(isEqualInteger(values, names, "BinCountNeg", binCountNeg1, binCountNeg2));
        Assert.assertTrue(isEqualDouble(values, names, "BinWeightedCountPos", binWeightedCountPos1, binWeightedCountPos2));
        Assert.assertTrue(isEqualDouble(values, names, "BinWeightedCountNeg", binWeightedCountNeg1, binWeightedCountNeg2));
        Assert.assertTrue(isEqualDouble(values, names, "BinPosRate", binPosRate1, binPosRate1));
        
    }
    
    private boolean isEqualInteger(List<String> values, List<String> names, String listname, List<Integer> list1, List<Integer> list2) {
        return values.get(names.indexOf(listname)).equals(list1.toString()) || values.get(names.indexOf(listname)).equals(list2.toString());
    }
    
    private boolean isEqualDouble(List<String> values, List<String> names, String listname, List<Double> list1, List<Double> list2) {
        return values.get(names.indexOf(listname)).equals(list1.toString()) || values.get(names.indexOf(listname)).equals(list2.toString());
    }
    
}
