package ml.shifu.shifu.core.stability;


import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.junit.Assert;
import org.junit.Test;

public class ChaosFactoryTest {
    @Test
    public void testGetChaosType() {
        Environment.setProperty(Constants.CHAOS_TYPE, "null");
        Environment.setProperty(Constants.CHAOS_COLUMNS, "");
        Assert.assertEquals(ChaosFactory.getInstance().getChaosType().getName(), ChaosType.NULL_VALUE.getName());
    }
}