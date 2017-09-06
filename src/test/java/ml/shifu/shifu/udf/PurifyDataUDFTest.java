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

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.JSONUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Writer;

/**
 * PurifyDataUDF class
 */
public class PurifyDataUDFTest {

    private PurifyDataUDF instanceA;
    private PurifyDataUDF instanceB;
    private PurifyDataUDF instanceC;
    @SuppressWarnings("unused")
    private PurifyDataUDF instanceD;

    @BeforeClass
    public void setUp() throws Exception {
        instanceC = new PurifyDataUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json", "EvalA");

        instanceB = new PurifyDataUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");

        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json", SourceType.LOCAL);

        modelConfig.getDataSet().setFilterExpressions("diagnosis == \"B\"");
        modelConfig.getEvalConfigByName("EvalA").getDataSet().setFilterExpressions("column_11");

        Writer writer = ShifuFileUtils.getWriter("ModelConfig.json", SourceType.LOCAL);
        JSONUtils.writeValue(writer, modelConfig);
        writer.close();

        instanceA = new PurifyDataUDF("LOCAL", "ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        instanceD = new PurifyDataUDF("LOCAL", "ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json", "EvalA");
    }

    @Test
    public void testEval() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for(int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "B");

        Assert.assertTrue(instanceA.exec(input));
        Assert.assertTrue(instanceB.exec(input));
        Assert.assertTrue(instanceC.exec(input));
        // Assert.assertFalse(instanceD.exec(input));

        input.set(0, "M");
        Assert.assertFalse(instanceA.exec(input));
        Assert.assertTrue(instanceB.exec(input));
        Assert.assertTrue(instanceC.exec(input));
        // Assert.assertFalse(instanceD.exec(input));
    }

    @AfterClass
    public void tearDown() throws IOException {
        ShifuFileUtils.deleteFile("ModelConfig.json", SourceType.LOCAL);
    }
}
