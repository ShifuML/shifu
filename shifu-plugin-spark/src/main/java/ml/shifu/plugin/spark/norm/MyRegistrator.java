/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.plugin.spark.norm;

import org.apache.spark.serializer.KryoRegistrator;
import com.esotericsoftware.kryo.Kryo;

/**
 * Registers the classes to be normalized, so that the full class name does not have to be serialized along 
 * with every object.
 */

public class MyRegistrator implements KryoRegistrator {

    public MyRegistrator() {
        // TODO Auto-generated constructor stub
    }

    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(BroadcastVariables.class);
    }

}
