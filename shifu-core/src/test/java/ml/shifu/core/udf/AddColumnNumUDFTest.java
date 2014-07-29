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
package ml.shifu.core.udf;

import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;


/**
 * AddColumnNumUDFTest class
 */
public class AddColumnNumUDFTest {

    private AddColumnNumUDF instance;

    @BeforeClass
    public void setUp() throws Exception {
        instance = new AddColumnNumUDF("LOCAL",
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json",
                "false");
    }

    // @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));
    }

    //@Test(expectedExceptions = ShifuException.class)
    public void testUDFNotEnoughInput() throws Exception {
        Tuple tuple = TupleFactory.getInstance().newTuple(20);

        tuple.set(0, "M");
        for (int i = 1; i < 20; i++) {
            tuple.set(i, 0.0d);
        }

        //Assert.assertNull(instance.exec(tuple));
        DataBag dataBag = instance.exec(tuple);
        Assert.assertEquals(19, dataBag.size());
    }

    // @Test
    public void testUDFEnoughInput() throws Exception {
        Tuple tuple = TupleFactory.getInstance().newTuple(32);

        for (int i = 0; i < 32; i++) {
            tuple.set(i, 0);
        }
        tuple.set(0, "123456");
        tuple.set(1, "M");

        DataBag dataBag = instance.exec(tuple);
        Assert.assertEquals(dataBag.size(), 30);

        Assert.assertEquals(dataBag.iterator().next().size(), 4);
    }
}
