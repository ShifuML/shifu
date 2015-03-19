package ml.shifu.shifu.core;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ConvergeJudgerTest {

    @Test
    public void testReset() {
        ConvergeJudger judger = new ConvergeJudger();
        judger.setTrainErr(1.0);
        judger.setTestErr(2.0);
        judger.setThreshold(3.0);
        
        judger.reset();
        Assert.assertTrue(judger.getTrainErr().compareTo(Double.POSITIVE_INFINITY) == 0);
        Assert.assertTrue(judger.getTestErr().compareTo(Double.POSITIVE_INFINITY) == 0);
        Assert.assertTrue(judger.getCurrentAvgErr().compareTo(Double.POSITIVE_INFINITY) == 0);
        Assert.assertTrue(judger.getThreshold().compareTo(Double.valueOf(0.0)) == 0);
    }
    
    @Test
    public void testIsConverged() {
        double train_err = 1.0;
        double test_err = 3.0;
        double threshold1 = 1.0;
        double threshold2 = 2.0;
        double threshold3 = 3.0;
        
        ConvergeJudger judger = new ConvergeJudger();
        judger.setTrainErr(train_err);
        judger.setTestErr(test_err);
        
        judger.setThreshold(threshold1);
        Assert.assertFalse(judger.isConverged());
        
        judger.setThreshold(threshold2);
        Assert.assertTrue(judger.isConverged());
        
        judger.setThreshold(threshold3);
        Assert.assertTrue(judger.isConverged());
        
        judger.setTrainErr(null);
        judger.setTestErr(0.0);
        Assert.assertFalse(judger.isConverged());
        Assert.assertTrue(judger.getCurrentAvgErr().compareTo(Double.POSITIVE_INFINITY) == 0);
        
        judger.setTrainErr(0.0);
        judger.setTestErr(null);
        Assert.assertFalse(judger.isConverged());
        Assert.assertTrue(judger.getCurrentAvgErr().compareTo(Double.POSITIVE_INFINITY) == 0);
        
        judger.setTestErr(0.00001);
        judger.setThreshold(null);
        Assert.assertFalse(judger.isConverged());
    }
}
