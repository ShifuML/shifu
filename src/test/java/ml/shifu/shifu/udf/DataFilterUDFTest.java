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
 * DataFilterUDFTest class
 */
public class DataFilterUDFTest {

    private DataFilterUDF instanceA;
    private DataFilterUDF instanceB;

    @BeforeClass
    public void setUp() throws Exception {
        instanceA = new DataFilterUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "0.0",
                "false");
        instanceB = new DataFilterUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "1.0",
                "false");
    }

    @Test
    public void testExec() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "M");

        Assert.assertNull(instanceA.exec(input));
        Assert.assertEquals(31, instanceB.exec(input).size());
    }
}
