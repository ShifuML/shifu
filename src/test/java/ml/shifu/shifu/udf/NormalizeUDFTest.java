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

import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
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
        instance = new NormalizeUDF("LOCAL",
                "src/test/resources/unittest/ModelSets/full/ModelConfig.json",
                "src/test/resources/unittest/ModelSets/full/ColumnConfig.json");
    }

    @Test
    public void testUDFNull() throws Exception {
        Assert.assertNull(instance.exec(null));

        Tuple tuple = TupleFactory.getInstance().newTuple(0);
        Assert.assertNull(instance.exec(tuple));
    }

    @Test
    public void testExec() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(32);
        for (int i = 0; i < 32; i++) {
            input.set(i, 1);
        }
        input.set(1, "M");

//        Assert.assertEquals(32, instance.exec(input).size());
//        Assert.assertEquals("(1,-3.713647,-4,-3.724801,-1.845747,4,4,4,4,4,4,2.286182,-0.365762,-0.925409,-0.864187,4,4,4,4,4,4,-3.147345,-4,-3.141658,-1.519785,4,4,3.481698,4,4,4,1.0)", instance.exec(input).toString());
    }

    //@Test
    public void testNegativeScore() throws IOException {
        String data = "B|11.75|17.56|75.89|422.9|0.1073|0.09713|0.05282|0.0444|0.1598|0.06677|0.4384|1.907|3.149|30.66|0.006587|0.01815|0.01737|0.01316|0.01835|0.002318|13.5|27.98|88.52|552.3|0.1349|0.1854|0.1366|0.101|0.2478|0.07757";
        String[] fields = data.split("\\|");

        Tuple input = TupleFactory.getInstance().newTuple(fields.length);
        for (int i = 0; i < fields.length; i++) {
            input.set(i, fields[i]);
        }

        Assert.assertEquals(32, instance.exec(input).size());
        Assert.assertEquals("0|-0.666117|-0.306386|-0.654091|-0.649112|0.720215|-0.16167|-0.443343|-0.121902|-0.747095|0.540368|0.173526|1.557082|0.184375|-0.19|-0.129016|-0.43034|-0.517333|0.231134|-0.27143|-0.611124|-0.55636|0.468762|-0.543583|-0.557313|0.104202|-0.455515|-0.652425|-0.217644|-0.692613|-0.358077|1.0", instance.exec(input).toDelimitedString("|"));
    }

    @Test
    public void testGetSchema() {
        Assert.assertEquals(instance.outputSchema(null).toString(), "{Normalized: (diagnosis: int,mean_radius: float,mean_texture: float,mean_perimeter: float,mean_area: float,mean_smoothness: float,mean_compactness: float,mean_concavity: float,mean_concave_points: float,mean_symmetry: float,mean_fractal_dimension: float,std_radius: float,std_texture: float,std_perimeter: float,std_area: float,std_smoothness: float,std_compactness: float,std_concavity: float,std_concave_points: float,std_symmetry: float,std_fractal_dimension: float,worst_radius: float,worst_texture: float,worst_perimeter: float,worst_area: float,worst_smoothness: float,worst_compactness: float,worst_concavity: float,worst_concave_points: float,worst_symmetry: float,worst_fractal_dimension: float,weight: float)}");
    }
}
