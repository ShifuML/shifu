package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.core.dtrain.nn.ActivationSwish;
import org.junit.Assert;
import org.testng.annotations.Test;

public class ActivationSwishTest {

    @Test
    public void testDerive() {
        ActivationSwish swish = new ActivationSwish();


        //Test activtion function
        double[] swishInputValue = {-1d,0,1d};
        double[] truth = {-1/(1+Math.exp(1)),0,1/(1+Math.exp(-1))};
        double error=0;

        swish.activationFunction(swishInputValue,0,3);
        for (int i=0;i<3;i++) {
            error += Math.abs(truth[i] - swishInputValue[i]);
        }
        Assert.assertTrue(error<1E-6);

        //Test derivative function
        double d0 = swish.derivativeFunction(0.0, 0.0);
        Assert.assertTrue(Math.abs(d0 - 0.5) < 1E-6);

        double d1 = swish.derivativeFunction(1, 0.0);
        double d11 = (1 + 2*Math.exp(-1))/Math.pow(1+Math.exp(-1),2);
        Assert.assertTrue(Math.abs(d1 - d11) < 1E-6);

        double dn1 = swish.derivativeFunction(-1, 0.0);
        double dn11 = 1/Math.pow(1+Math.exp(1),2);
        Assert.assertTrue(Math.abs(dn1 - dn11) < 1E-6);
    }
}
