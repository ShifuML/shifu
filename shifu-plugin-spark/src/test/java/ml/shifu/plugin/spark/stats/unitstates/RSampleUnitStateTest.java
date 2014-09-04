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

import junit.framework.Assert;

import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class RSampleUnitStateTest {
    RSampleUnitState<Integer> state= new RSampleUnitState<Integer>(500);
    @BeforeClass
    public void populate() {
        for(int i=1; i <= 1000; i++) {
            state.addSample(i);
        }
    }
    
    @Test
    public void testFullSample() {
        RSampleUnitState<Integer> fullState= new RSampleUnitState<Integer>(100);
        for(int i=1; i <= 100; i++)
            fullState.addSample(i);
        Assert.assertEquals(fullState.getSamples().size(), 100);
        for(int i=1; i < 100; i++)
            Assert.assertTrue(fullState.getSamples().contains(i));
    }
    
    @Test
    public void checkSample() {
        List<Integer> sample= state.getSamples();
        // check mean of sample
        int sum= 0;
        for(Integer value: sample)
            sum+= value;
        // check if mean is within 2% of actual mean
        System.out.println("mean is " + (double)sum/500);
        Assert.assertTrue(isWithin((double)sum/500, 500, 0.1));
        Assert.assertEquals(state.getSamples().size(), 500);
    }
    
    @Test
    public void testMerge() throws Exception {
        RSampleUnitState<Integer> state2= new RSampleUnitState<>(500);
        for(int i=1001; i <= 2000; i++)
            state2.addSample(i);
        state2.merge(state);
        int sum= 0;
        for(Integer value: state2.getSamples())
            sum+= value;
        // check if mean is within 2% of actual mean
        System.out.println("mean is " + (double)sum/500);
        Assert.assertEquals(state2.getSamples().size(), 500);
        Assert.assertTrue(isWithin((double)sum/500, 1000, 0.1));
    }

    private boolean isWithin(double testValue, double trueValue, double ratio) {
        return (testValue < trueValue*(1+ratio)) && (testValue > trueValue*(1-ratio));
    }
    
}
