package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.core.dtrain.wdl.activation.Activation;
import ml.shifu.shifu.core.dtrain.wdl.activation.ActivationFactory;
import ml.shifu.shifu.core.dtrain.wdl.activation.Sigmoid;
import org.testng.annotations.Test;

/**
 * @author haillu
 */
public class ActivationListTest  {
    @Test
    public void testActivationList(){
        ActivationFactory factory = ActivationFactory.getInstance();
        Activation acti = factory.getActivation("Sigmoid");
//        Activation acti = new Sigmoid();
//        System.out.println(acti.getClass());

    }
}
