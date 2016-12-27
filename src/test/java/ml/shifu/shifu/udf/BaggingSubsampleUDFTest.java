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
package ml.shifu.shifu.udf;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.io.FileUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class BaggingSubsampleUDFTest {

    private ObjectMapper jsonMapper = new ObjectMapper();

    @BeforeClass
    public void setUp() throws Exception {

        File file = new File("udf");
        if (!file.exists()) {
            FileUtils.forceMkdir(file);
        }

        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                SourceType.LOCAL);

        modelConfig.getTrain().setBaggingNum(1);
        ;
        modelConfig.getTrain().setBaggingSampleRate(2.0);
        ;

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(
                new File("udf/ModelConfig.json"),
                modelConfig);

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(
                new File("udf/ColumnConfig.json"),
                columnConfigList);
    }

    @Test
    public void test() throws Exception {
        BaggingSubsampleUDF instance = new BaggingSubsampleUDF("LOCAL",
                "udf/ModelConfig.json",
                "udf/ColumnConfig.json");

        Tuple tuple = TupleFactory.getInstance().newTuple();

        tuple.append("1");

        Assert.assertEquals("{(0,(1))}", instance.exec(tuple).toString());

        Assert.assertNull(instance.outputSchema(new Schema()));
    }

    @AfterClass
    public void delete() throws IOException {
        FileUtils.deleteDirectory(new File("udf"));
    }
}
