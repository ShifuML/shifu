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

import java.io.IOException;

import ml.shifu.shifu.util.Constants;
import org.apache.hadoop.conf.Configuration;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.util.UDFContext;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import ml.shifu.shifu.udf.norm.PrecisionType;

/**
 * NormalizeUDFTest class
 */
public class NormalizeUDFTest {

    private NormalizeUDF instance;

    @BeforeClass
    public void setUp() throws Exception {
        UDFContext.getUDFContext().addJobConf(new Configuration());
        instance = new NormalizeUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");
        instance.setPrecisionType(PrecisionType.FLOAT7);
    }

    @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    @Test
    public void testExec() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for(int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "M");
        input.set(1, "2.1");

        Assert.assertEquals(32, instance.exec(input).size());
        Assert.assertEquals(
                "(1,-3.3745382,-4.0,-3.697376,-1.8706726,4.0,4.0,4.0,4.0,4.0,4.0,2.473354,-0.3504254,-1.0068849,-1.0734632,4.0,4.0,4.0,4.0,4.0,4.0,-3.1432278,-4.0,-3.127431,-1.5752382,4.0,4.0,3.485806,4.0,4.0,4.0,2.1)",
                instance.exec(input).toString());
    }

    @Test
    public void testNegativeScore() throws IOException {
        String data = "B|11.75|17.56|75.89|422.9|0.1073|0.09713|0.05282|0.0444|0.1598|0.06677|0.4384|1.907|3.149|30.66|0.006587|0.01815|0.01737|0.01316|0.01835|0.002318|13.5|27.98|88.52|552.3|0.1349|0.1854|0.1366|0.101|0.2478|0.07757";
        String[] fields = data.split("\\|");

        Tuple input = TupleFactory.getInstance().newTuple(fields.length);
        for(int i = 0; i < fields.length; i++) {
            input.set(i, fields[i]);
        }

        Assert.assertEquals(32, instance.exec(input).size());
        Assert.assertEquals(
                "0|-0.66922235|-0.36015487|-0.65554094|-0.6652448|0.76039577|-0.13163291|-0.42296025|-0.106827006|-0.77677613|0.6052514|0.18077102|1.3659062|0.18749943|-0.22420046|-0.113538824|-0.40528342|-0.49631938|0.24349733|-0.26379234|-0.64114404|-0.57022554|0.43512848|-0.5549224|-0.5863008|0.11923738|-0.44027883|-0.62061524|-0.2076809|-0.70519555|-0.3397934|11.75",
                instance.exec(input).toDelimitedString("|"));
    }

    @Test
    public void testGetSchema() {
        Assert.assertEquals(instance.outputSchema(null).toString(),
                "{Normalized: (diagnosis: double,column_3: float,column_4: float,column_5: float,column_6: float,column_7: float,column_8: float,column_9: float,column_10: float,column_11: float,column_12: float,column_13: float,column_14: float,column_15: float,column_16: float,column_17: float,column_18: float,column_19: float,column_20: float,column_21: float,column_22: float,column_23: float,column_24: float,column_25: float,column_26: float,column_27: float,column_28: float,column_29: float,column_30: float,column_31: float,column_32: float,shifu::weight: float)}"
        );
    }

    @Test
    public void testCompactNorm() throws Exception {
        Configuration conf = new Configuration();
        // Norm only selected columns.
        conf.set(Constants.SHIFU_NORM_ONLY_SELECTED, "true");
        UDFContext.getUDFContext().addJobConf(conf);
        // The ColumnConfigCompact.json contains 31 columns. Only 3 columns are selected.
        NormalizeUDF instance2 = new NormalizeUDF("LOCAL",
            "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
            "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfigCompact.json");
        Tuple input = TupleFactory.getInstance().newTuple(31);
        for (int i = 0; i < 31; i++) {
            input.set(i, 1);
        }
        input.set(0, "M");
        // Set weight column's value.
        input.set(1, "2.1");

        Tuple output = instance2.exec(input);
        // Target, 3 selected columns, and weight, are 5 in total.
        Assert.assertEquals(5, output.size());
        Assert.assertEquals("(1,-3.3745382,-4.0,-3.697376,2.1)", output.toString());
    }

    @Test
    public void testColumnConfigCache() throws Exception {
        String modelConfigPath = "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json";
        String columnConfigPath = "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json";
        String compactColumnConfigPath = "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfigCompact.json";
        NormalizeUDF normUDF1 = new NormalizeUDF("LOCAL", modelConfigPath, columnConfigPath);
        NormalizeUDF normUDF2 = new NormalizeUDF("LOCAL", modelConfigPath, columnConfigPath);
        NormalizeUDF normUDF3 = new NormalizeUDF("LOCAL", modelConfigPath, compactColumnConfigPath);
        Assert.assertTrue(normUDF1.columnConfigList == normUDF2.columnConfigList);
        Assert.assertFalse(normUDF1.columnConfigList == normUDF3.columnConfigList);
    }
}
