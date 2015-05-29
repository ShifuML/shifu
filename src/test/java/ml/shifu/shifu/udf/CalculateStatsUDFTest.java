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

import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;

/**
 * CalculateStatsUDFTest class
 */
public class CalculateStatsUDFTest {

    private CalculateStatsUDF instance;

    @BeforeClass
    public void setUp() throws Exception {
        instance = new CalculateStatsUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json", "false");
    }

    @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    public void testExec() throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple(2);
        tuple.set(0, 5);

        DataBag dataBag = BagFactory.getInstance().newDefaultBag();
        for(int i = 0; i < 20; i++) {
            Tuple scoreTuple = TupleFactory.getInstance().newTuple(3);
            scoreTuple.set(0, i * 5);
            scoreTuple.set(1, i % 2);
            scoreTuple.set(2, 2);

            dataBag.add(scoreTuple);
        }
        tuple.set(1, dataBag);

        Assert.assertEquals(19, instance.exec(tuple).size());
        // Assert.assertEquals("(5,[-Infinity, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[NaN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],-1,-1,95,0,47.5,29.580399,N)",
        // instance.exec(tuple).toString());
    }
}
