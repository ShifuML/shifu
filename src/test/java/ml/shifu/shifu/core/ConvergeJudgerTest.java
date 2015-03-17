package ml.shifu.shifu.core;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ConvergeJudgerTest {

    @Test
    public void testIsConverged() {
        double train_err = 1.0;
        double test_err = 3.0;
        double threshold1 = 1.0;
        double threshold2 = 2.0;
        double threshold3 = 3.0;
        
        Assert.assertFalse(ConvergeJudger.isConverged(train_err, test_err, threshold1));
        Assert.assertTrue(ConvergeJudger.isConverged(train_err, test_err, threshold2));
        Assert.assertTrue(ConvergeJudger.isConverged(train_err, test_err, threshold3));
        
    }
}
