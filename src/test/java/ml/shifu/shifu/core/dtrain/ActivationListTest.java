package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.core.dtrain.wdl.activation.Activation;
import ml.shifu.shifu.core.dtrain.wdl.activation.ActivationFactory;
import ml.shifu.shifu.core.dtrain.wdl.activation.Sigmoid;
import org.apache.commons.collections.map.HashedMap;
import org.testng.annotations.Test;
import scala.Int;

import java.util.Map;

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

    @Test
    @SuppressWarnings("unchecked")
    public void testMapArray(){
        Map<String, Integer>[] maps = new Map[5];
        maps[0] = new HashedMap(){
            {
                put("adsfs",1);
            }
        };

        Map<String,Integer>[] realMaps;
        realMaps = maps;
        System.out.println(realMaps[0]);

    }
}
