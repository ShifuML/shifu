package ml.shifu.shifu.core;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class ConvergeJudgerTest {

    private ConvergeJudger judger;

    @BeforeClass
    public void setUp() {
        this.judger = new ConvergeJudger();
    }

    @Test
    public void testJudge() {
        Assert.assertTrue(judger.judge(1.0, 2.0));
        Assert.assertTrue(judger.judge(1.0, 1.0));
        Assert.assertFalse(judger.judge(1.0, 0.1));
    }

}
