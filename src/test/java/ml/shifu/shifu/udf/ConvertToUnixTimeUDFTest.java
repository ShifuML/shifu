/**
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.udf;

import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.testng.Assert;


public class ConvertToUnixTimeUDFTest {


    //@Test
    public void test1() throws ExecException {
        ConvertToUnixTimeUDF timer = new ConvertToUnixTimeUDF();

        Assert.assertNull(timer.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append("10/21/2013 18:16:32");

        Assert.assertEquals(timer.exec(tuple), Long.valueOf(1382350592));

        Assert.assertNull(timer.outputSchema(new Schema()));
    }

    //@Test
    public void test2() throws Exception {
        ConvertToUnixTimeUDF timer = new ConvertToUnixTimeUDF("yyyy/MM/dd HH:mm:ss");

        Assert.assertNull(timer.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append("2013/10/21 18:16:32");

        Assert.assertEquals(timer.exec(tuple), Long.valueOf(1382350592));

        Assert.assertNull(timer.outputSchema(new Schema()));
    }
}
