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

import ml.shifu.core.util.Environment;
import org.apache.commons.io.FileUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;


/**
 * EvalScoreUDFTest class
 */
public class EvalScoreUDFTest {

    private EvalScoreUDF instance;
    private File tmpModels = new File("models");

    @BeforeClass
    public void setUp() throws Exception {
        Environment.setProperty(Environment.SHIFU_HOME, ".");
        File models = new File("src/test/resources/unittest/ModelSets/full/models");

        FileUtils.copyDirectory(models, tmpModels);

        instance = new EvalScoreUDF("LOCAL",
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json",
                "Eval1");
    }

    @Test
    public void testUDFNull() throws Exception {
        //Assert.assertNull(instance.exec(null));
        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    //@Test
    public void testExec() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "M");

        Assert.assertEquals("(M,1.0,43,74,6,36,36,32,74,69,6)", instance.exec(input).toString());
    }

    //@Test
    public void testBelowScore() throws IOException {
        String data = "123456|B|13.87|20.7|89.77|584.8|0.09578|0.1018|0.03688|0.02369|0.162|0.06688|0.272|1.047|2.076|23.12|0.006298|0.02172|0.02615|0.009061|0.0149|0.003599|15.05|24.75|99.17|688.6|0.1264|0.2037|0.1377|0.06845|0.2249|0.08492";
        String[] fields = data.split("\\|");

        Tuple input = TupleFactory.getInstance().newTuple(fields.length);
        for (int i = 0; i < fields.length; i++) {
            input.set(i, fields[i]);
        }

        Assert.assertEquals("(B,1.0,157,195,130,152,195,152,135,173,130)", instance.exec(input).toString());
    }

    @Test
    public void testGetSchema() {
        Assert.assertEquals("{EvalScore: (core::diagnosis: chararray,core::weight: chararray,core::mean: int,core::max: int,core::min: int,core::median: int,core::model0: int,core::model1: int,core::model2: int,core::model3: int,core::model4: int)}", instance.outputSchema(null).toString());
    }

    @AfterClass
    public void clearUp() throws IOException {
        FileUtils.deleteDirectory(tmpModels);
    }
}
