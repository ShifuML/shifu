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

import org.dmg.pmml.Counts;
import org.dmg.pmml.UnivariateStats;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class FrequencyUnitStateTest {
    FrequencyUnitState state= new FrequencyUnitState();
    
    @BeforeClass
    public void populate() {
        for(int i=0; i < 10; i++)
            state.addData(1.0);
        for(int i=0; i < 10; i++)
            state.addData("");
        for(int i=0; i < 10; i++)
            state.addData("hello");
        for(int i=0; i < 10; i++)
            state.addData(null);        
    }
    
    @Test
    public void testInvalid() {
        Assert.assertEquals(state.getInvalid(), 10.0);
    }

    @Test
    public void testMissing() {
        Assert.assertEquals(state.getMissing(), 20.0);
    }

    @Test
    public void testTotal() {
        Assert.assertEquals(state.getTotal(), 40.0);
    }
    
    @Test 
    public void testPMML() {
        UnivariateStats s= new UnivariateStats();
        state.populateUnivariateStats(s, null);
        Counts c= s.getCounts();
        Assert.assertEquals(c.getInvalidFreq(), 10.0);
        Assert.assertEquals(c.getTotalFreq(), 40.0);
        Assert.assertEquals(c.getMissingFreq(), 20.0);
    }
}
