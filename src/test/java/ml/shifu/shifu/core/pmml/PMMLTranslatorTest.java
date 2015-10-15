/*
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
package ml.shifu.shifu.core.pmml;

import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.util.Environment;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * PMMLTranslatorTest class
 */
public class PMMLTranslatorTest {

    @BeforeClass
    public void setUp() {
        Environment.setProperty(Environment.SHIFU_HOME, ".");
    }

    // @Test
    public void testAllNumericVariablePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("cancer-judgement",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models", 5, "EvalA",
                "src/test/resources/example/cancer-judgement/DataStore/Full_data/data.dat", "\\|", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }

    // @Test
    public void testWoeVariablePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("testWoePmml",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet2/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet2/ColumnConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet2/models", 5, "EvalA",
                "src/test/resources/example/cancer-judgement/DataStore/Full_data/data.dat", "\\|", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }

    // @Test
    public void testMixTypeVariablePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("ModelK",
                "src/test/resources/example/labor-neg/DataStore/DataSet1/ModelConfig.json",
                "src/test/resources/example/labor-neg/DataStore/DataSet1/ColumnConfig.json",
                "src/test/resources/example/labor-neg/DataStore/DataSet1/models", 1, "EvalA",
                "src/test/resources/example/labor-neg/DataStore/DataSet1/data.dat", ",", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }

    @Test
    public void testLRNumericVariablePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("cancer-judgement",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/LR/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/LR/ColumnConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/LR/models",
                ModelTrainConf.ALGORITHM.LR, 1, "EvalA",
                "src/test/resources/example/cancer-judgement/DataStore/Full_data/data.dat", "\\|", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }

    // @Test
    public void testMixTypeWoePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("testWoe2Pmml",
                "src/test/resources/example/labor-neg/DataStore/DataSet2/ModelConfig.json",
                "src/test/resources/example/labor-neg/DataStore/DataSet2/ColumnConfig.json",
                "src/test/resources/example/labor-neg/DataStore/DataSet2/models", 5, "EvalA",
                "src/test/resources/example/labor-neg/DataStore/DataSet1/data.dat", ",", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }

    // @Test
    public void testWoeZscorePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("TestWoeZscale",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet3/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet3/ColumnConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet3/models", 5, "Eval1",
                "src/test/resources/example/cancer-judgement/DataStore/Full_data/data.dat", "\\|", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }

    // @Test
    public void testMixTypeWoeZscorePmmlCase() throws Exception {
        PMMLVerifySuit verifySuit = new PMMLVerifySuit("TestMixTypeWoeZscale",
                "src/test/resources/example/labor-neg/DataStore/ModelSet3/ModelConfig.json",
                "src/test/resources/example/labor-neg/DataStore/ModelSet3/ColumnConfig.json",
                "src/test/resources/example/labor-neg/DataStore/ModelSet3/models", 5, "EvalA",
                "src/test/resources/example/labor-neg/DataStore/DataSet1/data.dat", ",", 1.0d, true);
        Assert.assertTrue(verifySuit.doVerification());
    }
}
