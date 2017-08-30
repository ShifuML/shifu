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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;


/**
 * DataPurifierTest class
 */
public class DataPurifierTest {

    private DataPurifier dataPurifier;
    private ModelConfig modelConfig;

    @BeforeClass
    public void setUp() throws Exception {
        modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                SourceType.LOCAL);
    }

    @Test
    public void testIsFilterOutA() throws IOException {
        dataPurifier = new DataPurifier(modelConfig);
        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        modelConfig.getDataSet().setFilterExpressions("aaa == aaa");
        dataPurifier = new DataPurifier(modelConfig);
        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        modelConfig.getDataSet().setFilterExpressions("1 == 2");
        dataPurifier = new DataPurifier(modelConfig);
        Assert.assertFalse(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        modelConfig.getDataSet().setFilterExpressions("*");
        dataPurifier = new DataPurifier(modelConfig);
        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        EvalConfig evalConfig = modelConfig.getEvalConfigByName("EvalA");
        evalConfig.getDataSet().setFilterExpressions("diagnosis == \"M\"");
        dataPurifier = new DataPurifier(evalConfig);
        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));
        Assert.assertFalse(dataPurifier.isFilter("B|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        evalConfig.getDataSet().setFilterExpressions("  ");
        dataPurifier = new DataPurifier(evalConfig);
        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));
        Assert.assertTrue(dataPurifier.isFilter("B|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        evalConfig.getDataSet().setFilterExpressions(" ASDF *** SDFKSADFJKS >  ");
        dataPurifier = new DataPurifier(evalConfig);
        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));
        Assert.assertTrue(dataPurifier.isFilter("B|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));
    }

    @Test
    public void testFilterNull() throws IOException {
        modelConfig.getDataSet().setFilterExpressions("diagnosis != \"null\"");
        dataPurifier = new DataPurifier(modelConfig);
        Assert.assertFalse(dataPurifier.isFilter("null|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        Assert.assertTrue(dataPurifier.isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));
    }

    @Test
    public void testFilterEqualNull() throws IOException {
        modelConfig.getDataSet().setFilterExpressions("diagnosis != \"NULL\" ");
        dataPurifier = new DataPurifier(modelConfig);
        Assert.assertFalse(dataPurifier
                .isFilter("NULL|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));

        Assert.assertTrue(dataPurifier
                .isFilter("M|17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189"));
    }
    
    @Test
    public void testFilterIsNull() throws IOException {
        modelConfig.getDataSet().setFilterExpressions("diagnosis != null ");
        dataPurifier = new DataPurifier(modelConfig);
        
        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append(null);
        String[] fields = "17.99|10.38|122.8|1001|0.1184|0.2776|0.3001|0.1471|0.2419|0.07871|1.095|0.9053|8.589|153.4|0.006399|0.04904|0.05373|0.01587|0.03003|0.006193|25.38|17.33|184.6|2019|0.1622|0.6656|0.7119|0.2654|0.4601|0.1189".split("\\|");
        for ( String f : fields ) {
            tuple.append(f);
        }

        Assert.assertFalse(dataPurifier.isFilter(tuple));
        
        tuple = TupleFactory.getInstance().newTuple();
        tuple.append(new Object());
        for ( String f : fields ) {
            tuple.append(f);
        }
        
        Assert.assertTrue(dataPurifier.isFilter(tuple));

        modelConfig.getDataSet().setFilterExpressions("diagnosis == null ");
        dataPurifier = new DataPurifier(modelConfig);
        
        tuple = TupleFactory.getInstance().newTuple();
        tuple.append(null);
        for ( String f : fields ) {
            tuple.append(f);
        }

        Assert.assertTrue(dataPurifier.isFilter(tuple));
        
        tuple = TupleFactory.getInstance().newTuple();
        tuple.append(new Object());
        for ( String f : fields ) {
            tuple.append(f);
        }
        
        Assert.assertFalse(dataPurifier.isFilter(tuple));
    }
}
