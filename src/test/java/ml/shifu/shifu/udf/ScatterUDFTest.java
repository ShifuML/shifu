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

import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;

/**
 * ScatterUDFTest class
 */
public class ScatterUDFTest {

    private ScatterUDF instance;

    @BeforeClass
    public void setUp() throws Exception {
        instance = new ScatterUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
    }

    @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));
        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    @Test
    public void testExec() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(32);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }

        input.set(31, 0.8);

        Assert.assertEquals(30, instance.exec(input).size());
        Assert.assertEquals(3, instance.exec(input).iterator().next().size());
        Assert.assertEquals(0.8, instance.exec(input).iterator().next().get(2));
    }

}
