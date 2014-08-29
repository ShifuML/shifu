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
