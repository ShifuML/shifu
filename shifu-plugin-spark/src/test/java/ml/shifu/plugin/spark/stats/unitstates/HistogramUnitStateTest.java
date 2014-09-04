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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.StringUtils;
import org.dmg.pmml.Array;
import org.dmg.pmml.DiscrStats;
import org.dmg.pmml.UnivariateStats;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class HistogramUnitStateTest {
    HistogramUnitState state= new HistogramUnitState(100);
    
    @BeforeClass
    public void addData() {
        state.addData("a");
        state.addData("a");
        state.addData("a");
        state.addData(1);
        state.addData("1");
        state.addData(2.0);
    }
    
    @Test
    public void histogramTest() {
        Map<Object, Integer> m= state.getHistogram();
        Assert.assertEquals(m.keySet().size(), 4);
        Assert.assertEquals(m.get("a"), (Integer) 3);
        Assert.assertEquals(m.get(1), (Integer) 1);
        Assert.assertEquals(m.get("1"), (Integer) 1);
        Assert.assertEquals(m.get(2.0), (Integer) 1);
    }
    
    @Test
    public void mergeTest() throws Exception {
        HistogramUnitState state2= new HistogramUnitState(100);
        state2.addData("b");
        state2.addData("1");
        state2.merge(state);
        Map<Object, Integer> m= state2.getHistogram();
        Assert.assertEquals(m.keySet().size(), 5);
        Assert.assertEquals(m.get("a"), (Integer) 3);
        Assert.assertEquals(m.get(1), (Integer) 1);
        Assert.assertEquals(m.get("1"), (Integer) 2);
        Assert.assertEquals(m.get(2.0), (Integer) 1);
        Assert.assertEquals(m.get("b"), (Integer) 1);
    }
    
    @Test
    public void testMaxSize() {
        HistogramUnitState maxState= new HistogramUnitState(10);
        for(int i=0; i < 20; i++) 
            maxState.addData(i);
        Assert.assertEquals(maxState.getHistogram().size(), 10);
    }
    
    @Test
    public void testPMML() {
        HistogramUnitState pState= new HistogramUnitState(100);
        pState.addData("a");
        pState.addData("a");
        pState.addData("a");
        pState.addData("b");
        pState.addData("b");
        List<String> keys1= new ArrayList<String>();
        keys1.add("a");
        keys1.add("b");
        List<String> keys2= new ArrayList<String>();
        keys2.add("b");
        keys2.add("a");
        
        List<Integer> values1= new ArrayList<Integer>();
        values1.add(3);
        values1.add(2);
        List<Integer> values2= new ArrayList<Integer>();
        values2.add(2);
        values2.add(3);
        
        UnivariateStats stats= new UnivariateStats();
        pState.populateUnivariateStats(stats, null);
        DiscrStats discrStats= stats.getDiscrStats();
        List<Array> arrays= discrStats.getArrays();
        Assert.assertEquals(arrays.size(), 2);
        
        Assert.assertTrue((arrays.get(0).getValue().equals(StringUtils.join(values1, " ")) || arrays.get(0).getValue().equals(StringUtils.join(values2, " "))));
        Assert.assertEquals(arrays.get(0).getType(), Array.Type.INT);
        Assert.assertEquals(arrays.get(0).getN(), (Integer)2);
        
        Assert.assertTrue((arrays.get(1).getValue().equals(StringUtils.join(keys1, " ")) || arrays.get(1).getValue().equals(StringUtils.join(keys2, " "))));
        Assert.assertEquals(arrays.get(1).getType(), Array.Type.STRING);
        Assert.assertEquals(arrays.get(1).getN(), (Integer)2);
        
    }
    
}
