package ml.shifu.shifu.core;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ConvergeJudgerTest {

    @Test
    public void testJudge() {
        ConvergeJudger judger = new ConvergeJudger();
        
        Assert.assertTrue(judger.judge(1.0, 2.0));
        Assert.assertTrue(judger.judge(1.0, 1.0));
        Assert.assertFalse(judger.judge(1.0, 0.1));
    }
    
}
