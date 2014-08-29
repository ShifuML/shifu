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
