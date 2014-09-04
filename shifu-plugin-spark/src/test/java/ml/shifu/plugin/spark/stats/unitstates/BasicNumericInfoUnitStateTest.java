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

import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.BasicNumericInfoUnitState;

import org.dmg.pmml.ContStats;
import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.UnivariateStats;
import org.testng.annotations.Test;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;

public class BasicNumericInfoUnitStateTest {
    

    BasicNumericInfoUnitState state= new BasicNumericInfoUnitState();
    
    @BeforeClass
    public void populateStatetest() {
        state.addData((Double)1.0);
        state.addData((Double)2.0);
        state.addData((Double)3.0);
        state.addData((Double)4.0);
        state.addData((Double)5.0);
    }
    
    @Test
    public void minTest() {
        Assert.assertEquals(state.getMin(), 1.0);
    }
    
    @Test
    public void maxTest() {
        Assert.assertEquals(state.getMax(), 5.0);
    }
    
    @Test
    public void testSum() {
        Assert.assertEquals(state.getSum(), 15.0);
    }
    
    @Test
    public void testSumSqr() {
        Assert.assertEquals(state.getSumSqr(), 55.0);
    }
    
    @Test
    public void testN() {
        Assert.assertEquals(state.getN(), (Integer) 5);
    }
    
    @Test
    public void testPMML() {
        UnivariateStats us= new UnivariateStats();
        state.populateUnivariateStats(us, null);
        ContStats cs= us.getContStats();
        Assert.assertEquals(cs.getTotalSquaresSum(), 55.0);
        Assert.assertEquals(cs.getTotalValuesSum(), 15.0);
        NumericInfo ni= us.getNumericInfo();
        Assert.assertEquals(ni.getMaximum(), 5.0);
        Assert.assertEquals(ni.getMinimum(), 1.0);
        Assert.assertEquals(ni.getMean(), 3.0);
        //Assert.assertEquals(ni.getStandardDeviation(), )
    }
    
    @Test 
    public void nullTest() {
        state.addData(null);
        Assert.assertEquals(state.getN(), (Integer) 5);
    }
    
    @Test
    public void garbageTest() {
        state.addData("hello");
        Assert.assertEquals(state.getN(), (Integer)5);
    }
    
    @Test
    public void getNewBlankTest() {
        UnitState blank= state.getNewBlank();
        Assert.assertTrue(blank instanceof BasicNumericInfoUnitState);
        BasicNumericInfoUnitState newBlank= (BasicNumericInfoUnitState) blank;
        Assert.assertEquals(newBlank.getN(), (Integer) 0);
        Assert.assertEquals(newBlank.getMin(), Double.POSITIVE_INFINITY);
        Assert.assertEquals(newBlank.getMax(), Double.NEGATIVE_INFINITY);
        Assert.assertEquals(newBlank.getSum(), 0.0);
        Assert.assertEquals(newBlank.getSumSqr(), 0.0);
        
    }
    
    @Test 
    public void mergeTest() throws Exception {
        BasicNumericInfoUnitState state2= new BasicNumericInfoUnitState();
        for(int i=6; i <= 10; i++)
            state2.addData(i);
        state2.merge(state);
        Assert.assertEquals(state2.getN(), (Integer) 10);
        Assert.assertEquals(state2.getMax(), 10.0);
        Assert.assertEquals(state2.getMin(), 1.0);
        Assert.assertEquals(state2.getSum(), 55.0);
        Assert.assertEquals(state2.getSumSqr(), 385.0);
        
        
        UnivariateStats us= new UnivariateStats();
        state2.populateUnivariateStats(us, null);
        ContStats cs= us.getContStats();
        Assert.assertEquals(cs.getTotalSquaresSum(), 385.0);
        Assert.assertEquals(cs.getTotalValuesSum(), 55.0);
        NumericInfo ni= us.getNumericInfo();
        Assert.assertEquals(ni.getMaximum(), 10.0);
        Assert.assertEquals(ni.getMinimum(), 1.0);
        Assert.assertEquals(ni.getMean(), 5.5);
        //Assert.assertEquals(ni.getStandardDeviation(), 0);    
    }
    
}
