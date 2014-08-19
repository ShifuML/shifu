/*
 * Registers the classes to be normalized, so that the full class name does not have to be serialized along 
 * with every object.
 */
package ml.shifu.plugin.spark.norm;

import org.apache.spark.serializer.KryoRegistrator;
import com.esotericsoftware.kryo.Kryo;

public class MyRegistrator implements KryoRegistrator {

    public MyRegistrator() {
        // TODO Auto-generated constructor stub
    }

    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(BroadcastVariables.class);
    }

}
