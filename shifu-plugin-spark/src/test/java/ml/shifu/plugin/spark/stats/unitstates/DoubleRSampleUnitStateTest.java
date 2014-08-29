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
