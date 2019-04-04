package ml.shifu.shifu.core.binning;


import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Created by zhanhu on 5/8/17.
 */
public class ColumnConfigDynamicBinningTest {

    @Test
    public void testIvReduce() {
        double iv1 = calcualte(0, 28);
        double iv2 = calcualte(0, 30);
        double iv = calcualte(0, 58);
        Assert.assertTrue(iv1 + iv2 < iv);
    }

    public double calcualte(double cntP, double cntN) {
        double p = cntP / 1212895.0;
        double n = cntN / 1.1486916E7;

        double woePerBin = Math.log((p + EPS) / (n + EPS));
        return (p - n) * woePerBin;
    }

    private final static double EPS = 1e-10;
}
