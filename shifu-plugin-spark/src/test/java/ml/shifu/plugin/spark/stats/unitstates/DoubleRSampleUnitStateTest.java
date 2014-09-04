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

import java.util.List;

import ml.shifu.core.util.Params;

import org.dmg.pmml.NumericInfo;
import org.dmg.pmml.UnivariateStats;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class DoubleRSampleUnitStateTest {
    DoubleRSampleUnitState state= new DoubleRSampleUnitState(10);
    
    @BeforeClass
    public void addData() {
        for(int i=6; i < 10; i++)
            state.addData(i);
        state.addData("0");
        state.addData(1);
        state.addData("2");
        state.addData(3.0);
        state.addData((float)4);
        state.addData(5.0);
    }
    
    @Test
    public void testMedian() {
        Assert.assertEquals(state.getMedian(), 5.0);
    }
    
    @Test
    public void sortTest() {
        state.sortSamples();
        List<Double> s= state.getSamples();
        for(int i=0; i < 10; i++)
            Assert.assertEquals(s.get(i), (double)i);
    }
    
    @Test 
    public void testMax() {
        DoubleRSampleUnitState maxState= new DoubleRSampleUnitState(10);
        for(int i=0; i < 20; i++)
            maxState.addData(i);
        Assert.assertEquals(maxState.getSamples().size(), 10);
    }
    
    @Test 
    public void testPMML() {
        UnivariateStats stats= new UnivariateStats();
        Params params= new Params();
        params.put("numQuantiles", 2);
        state.populateUnivariateStats(stats, params);
        NumericInfo numInfo= stats.getNumericInfo();
        
        Assert.assertEquals(numInfo.getInterQuartileRange(), 5.0);
        Assert.assertEquals(numInfo.getMedian(), 5.0);
        // TODO: test quantiles
    }
}
