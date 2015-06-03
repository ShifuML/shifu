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

import ml.shifu.shifu.util.Environment;
import org.apache.commons.io.FileUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;

import java.io.File;
import java.io.IOException;


/**
 * CalculateReasonCodeUDFTest class
 */
public class CalculateReasonCodeUDFTest {

    private CalculateReasonCodeUDF instance;

    @BeforeClass
    public void setUp() throws Exception {
        File tmpCommon = new File("common");
        File common = new File("src/test/resources/common");
        FileUtils.copyDirectory(common, tmpCommon);

        Environment.setProperty(Environment.SHIFU_HOME, ".");
        instance = new CalculateReasonCodeUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "EvalA");
    }

    //@Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    //@Test( expectedExceptions = NumberFormatException.class)
    public void testExec() throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple(31);

        for (int i = 0; i < 31; i++) {
            tuple.set(i, i * 10);
        }
        tuple.set(0, "M");

        //TODO the CalculateReasonCodeUDF is not for common example, need to change
        Assert.assertEquals("", instance.exec(tuple).toString());
    }

    @AfterClass
    public void tearDown() throws IOException {
        FileUtils.deleteDirectory(new File("common"));
    }
}
