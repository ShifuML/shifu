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
import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
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
        File models = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");

        FileUtils.copyDirectory(models, tmpModels);

        instance = new EvalScoreUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "EvalA");
    }

    @Test
    public void testUDFNull() throws Exception {
        //Assert.assertNull(instance.exec(null));
        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

//    @Test
    public void testExec() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "M");

        check(instance.exec(input), StringUtils.split("M,1.0,42,74,5,35,35,30,74,65,5", ","));
    }

//    @Test
    public void testBelowScore() throws IOException {
        String data = "B|13.87|20.7|89.77|584.8|0.09578|0.1018|0.03688|0.02369|0.162|0.06688|0.272|1.047|2.076|23.12|0.006298|0.02172|0.02615|0.009061|0.0149|0.003599|15.05|24.75|99.17|688.6|0.1264|0.2037|0.1377|0.06845|0.2249|0.08492";
        String[] fields = data.split("\\|");

        Tuple input = TupleFactory.getInstance().newTuple(fields.length);
        for (int i = 0; i < fields.length; i++) {
            input.set(i, fields[i]);
        }

        check(instance.exec(input), StringUtils.split("B,1.0,7,11,3,8,6,8,11,3,8", ","));
    }

    @Test
    public void testGetSchema() {
        Assert.assertEquals("{EvalScore: (shifu::diagnosis: chararray,shifu::weight: chararray,shifu::mean: double,shifu::max: double,shifu::min: double,shifu::median: double,shifu::model0: double,shifu::model1: double,shifu::model2: double,shifu::model3: double,shifu::model4: double)}", instance.outputSchema(null).toString());
    }

//    @Test
    public void testExecScale() throws IOException {
        EvalScoreUDF scaleInst = new EvalScoreUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "EvalA", "1000000000");

        Tuple input = TupleFactory.getInstance().newTuple(31);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "M");

        check(scaleInst.exec(input), StringUtils.split("M,1.0,42431529,74243477,5347463,35996826,35996826,30754353,74243477,65815524,5347463", ","));
    }


    private void check(Tuple result, Object[] expectVals) throws ExecException {
        for ( int i = 0; i < expectVals.length; i ++ ) {
            Object obj = result.get(i);
            String value = obj.toString();
            if ( obj instanceof Double ) {
                Double actualVal = (Double) obj;
                value = Integer.toString(actualVal.intValue());
            }
            Assert.assertEquals(value, expectVals[i]);
        }
    }

    @AfterClass
    public void clearUp() throws IOException {
        FileUtils.deleteDirectory(tmpModels);
    }
}