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
package ml.shifu.shifu.udf;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.container.obj.ColumnConfig;
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
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json"
                );
    }

    @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    @Test
    public void testExec() throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple(2);
        tuple.set(0, 5);

        DataBag dataBag = BagFactory.getInstance().newDefaultBag();
        for ( int i = 0; i < 50; i ++ ) {
            Tuple scoreTuple = TupleFactory.getInstance().newTuple(3);
            scoreTuple.set(0, i % 5);
            scoreTuple.set(1, i % 3 ==0 ? "B": "M");
            scoreTuple.set(2, 2);

            dataBag.add(scoreTuple);
        }
        tuple.set(1, dataBag);

        ObjectMapper jsonMapper = new ObjectMapper();
        Assert.assertEquals(instance.exec(tuple).size(), 2);

        String s = instance.exec(tuple).get(1).toString();
        ColumnConfig config = jsonMapper.readValue(s, ColumnConfig.class);
        System.out.println(config.getBinBoundary());
        Assert.assertEquals(config.getBinBoundary().toString(), "[-Infinity, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]");
    }
}
