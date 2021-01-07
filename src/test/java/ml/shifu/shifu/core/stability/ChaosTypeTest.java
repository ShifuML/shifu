package ml.shifu.shifu.core.stability;

import ml.shifu.shifu.core.stability.algorithm.DeviationChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.NullValueChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.RandomChaosAlgorithm;
import org.junit.Assert;
import org.junit.Test;

public class ChaosTypeTest {
    @Test
    public void testGetChaosAlgorithm() {
        Assert.assertTrue(ChaosType.NULL_VALUE.getChaosAlgorithm() instanceof NullValueChaosAlgorithm);
        Assert.assertTrue(ChaosType.RANDOM_VALUE.getChaosAlgorithm() instanceof RandomChaosAlgorithm);
        Assert.assertTrue(ChaosType.DEVIATION_VALUE.getChaosAlgorithm() instanceof DeviationChaosAlgorithm);

    }

    @Test
    public void testFromName() {
        Assert.assertEquals(ChaosType.fromName("null").getName(), "null");
        Assert.assertNull(ChaosType.fromName("not_exist"));
    }
}