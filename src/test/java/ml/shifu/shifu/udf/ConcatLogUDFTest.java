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
import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;


/**
 * ConcatLogUDFTest class
 */
public class ConcatLogUDFTest {

    private ConcatLogUDF instance;

    @BeforeClass
    public void setUp() throws Exception {
        instance = new ConcatLogUDF();
    }

    @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    @Test
    public void testExec() throws ExecException {
        Tuple tuple = TupleFactory.getInstance().newTuple(1);

        DataBag dataBag = BagFactory.getInstance().newDefaultBag();
        for (int i = 0; i < 2; i++) {
            Tuple scoreTuple = TupleFactory.getInstance().newTuple(1);
            scoreTuple.set(0, "Hello, World! OK, let's start from 1234.");

            dataBag.add(scoreTuple);
        }
        tuple.set(0, dataBag);

        Assert.assertEquals("(5,Hello, World! OK, let's start from 1234.Hello, World! OK, let's start from 1234.)", instance.exec(tuple).toString());
    }
}
