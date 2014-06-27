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

import ml.shifu.core.container.obj.ModelConfig;
import ml.shifu.core.container.obj.RawSourceData.SourceType;
import ml.shifu.core.fs.ShifuFileUtils;
import ml.shifu.core.util.CommonUtils;
import ml.shifu.core.util.JSONUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;

import java.io.IOException;
import java.io.Writer;


/**
 * PurifyDataUDF class
 */
public class PurifyDataUDFTest {

    private PurifyDataUDF instanceA;
    private PurifyDataUDF instanceB;
    private PurifyDataUDF instanceC;
    private PurifyDataUDF instanceD;

    @BeforeClass
    public void setUp() throws Exception {
        instanceC = new PurifyDataUDF("LOCAL",
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json",
                "Eval1");

        instanceB = new PurifyDataUDF("LOCAL",
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json");

        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                SourceType.LOCAL);

        modelConfig.getDataSet().setFilterExpressions("diagnosis == \"B\"");
        modelConfig.getEvalConfigByName("Eval1").getDataSet().setFilterExpressions("column_11");

        Writer writer = ShifuFileUtils.getWriter("ModelConfig.json", SourceType.LOCAL);
        JSONUtils.writeValue(writer, modelConfig);
        writer.close();

        instanceA = new PurifyDataUDF("LOCAL",
                "ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json");
        instanceD = new PurifyDataUDF("LOCAL",
                "ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json",
                "Eval1");
    }

    //@Test
    public void testEval() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "B");

        Assert.assertTrue(instanceA.exec(input));
        Assert.assertTrue(instanceB.exec(input));
        Assert.assertTrue(instanceC.exec(input));
        Assert.assertFalse(instanceD.exec(input));

        input.set(0, "M");
        Assert.assertFalse(instanceA.exec(input));
        Assert.assertTrue(instanceB.exec(input));
        Assert.assertTrue(instanceC.exec(input));
        Assert.assertFalse(instanceD.exec(input));
    }

    @AfterClass
    public void tearDown() throws IOException {
        ShifuFileUtils.deleteFile("ModelConfig.json", SourceType.LOCAL);
    }
}
