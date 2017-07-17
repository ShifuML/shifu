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

import org.apache.hadoop.conf.Configuration;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.util.UDFContext;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;

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
                "(1,-3.374538,-4,-3.697376,-1.870673,4,4,4,4,4,4,2.473354,-0.350425,-1.006885,-1.073463,4,4,4,4,4,4,-3.143228,-4,-3.127431,-1.575238,4,4,3.485806,4,4,4,2.1)",
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
                "0|-0.669222|-0.360155|-0.655541|-0.665245|0.760396|-0.131633|-0.42296|-0.106827|-0.776776|0.605251|0.180771|1.365906|0.187499|-0.2242|-0.113539|-0.405283|-0.496319|0.243497|-0.263792|-0.641144|-0.570226|0.435128|-0.554922|-0.586301|0.119237|-0.440279|-0.620615|-0.207681|-0.705196|-0.339793|11.75",
                instance.exec(input).toDelimitedString("|"));
    }

    // @Test
    public void testGetSchema() {
        Assert.assertEquals(
                "{Normalized: (diagnosis: int,column_3: float,column_4: float,column_5: float,column_6: float,column_7: float,column_8: float,column_9: float,column_10: float,column_11: float,column_12: float,column_13: float,column_14: float,column_15: float,column_16: float,column_17: float,column_18: float,column_19: float,column_20: float,column_21: float,column_22: float,column_23: float,column_24: float,column_25: float,column_26: float,column_27: float,column_28: float,column_29: float,column_30: float,column_31: float,column_32: float,weight: float)}",
                instance.outputSchema(null).toString());
    }
}
