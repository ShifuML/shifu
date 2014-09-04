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

import ml.shifu.plugin.spark.stats.SerializedNumericalValueObject;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class BinomialRSampleUnitStateTest {

    BinomialRSampleUnitState state= new BinomialRSampleUnitState(100, 10);
    List<SerializedNumericalValueObject> nvoList;
    @BeforeClass
    public void addData() {
        nvoList= new ArrayList<>();
        
        nvoList.add(new SerializedNumericalValueObject((double)1, true, (double)1));
        nvoList.add(new SerializedNumericalValueObject((double)1, false, (double)1));
        nvoList.add(new SerializedNumericalValueObject((double)1, true, (double)0));
        nvoList.add(new SerializedNumericalValueObject((double)1, true, (double)-1));
        nvoList.add(new SerializedNumericalValueObject((double)1, false, (double)2));
        nvoList.add(new SerializedNumericalValueObject((double)2, true, (double)2));
        nvoList.add(new SerializedNumericalValueObject((double)2, false, (double)2));
        nvoList.add(new SerializedNumericalValueObject((double)2, true, (double)1));
        nvoList.add(new SerializedNumericalValueObject((double)2, false, (double)-1));
        nvoList.add(new SerializedNumericalValueObject((double)2, true, (double)0));
        for(SerializedNumericalValueObject nvo: nvoList)
            state.addData(nvo);
    }
    
    @Test
    public void testSample() {
        List<SerializedNumericalValueObject> m= state.getSamples();
        Assert.assertEquals(m.size(), 10);
        for(SerializedNumericalValueObject nvo:nvoList)
            Assert.assertTrue(m.contains(nvo));        
    }
    
    @Test
    public void testPMML() {
        // TODO
    }
}
