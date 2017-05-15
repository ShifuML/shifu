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

import java.io.IOException;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class VariableSelectorTest {

    ModelConfig modelConfig;
    List<ColumnConfig> columnConfigList;

    @BeforeClass
    public void setup() throws IOException {
        modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                SourceType.LOCAL);

        modelConfig.getVarSelect().setFilterNum(20);
        columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                SourceType.LOCAL);
    }

    @Test
    public void testFilter() throws IOException {
        modelConfig.getVarSelect().setFilterBy("ks");
        VariableSelector selector = new VariableSelector(modelConfig, columnConfigList);
        List<ColumnConfig> selected = selector.selectByFilter();

        Integer i = 0;

        for (ColumnConfig config : selected) {
            if (config.isFinalSelect()) i++;
        }
        Assert.assertEquals(i, modelConfig.getVarSelectFilterNum());

        modelConfig.getVarSelect().setFilterBy("iv");
        selector = new VariableSelector(modelConfig, columnConfigList);
        selector.selectByFilter();
        Assert.assertEquals(i, modelConfig.getVarSelectFilterNum());

        modelConfig.getVarSelect().setFilterBy("mix");
        selector = new VariableSelector(modelConfig, columnConfigList);
        selector.selectByFilter();
        Assert.assertEquals(i, modelConfig.getVarSelectFilterNum());
    }

//    @Test
//    public void testGetMSE() throws IOException {
//        Random random = new Random(new Date().getTime());
//
//        double[] input = new double[modelConfig.getVarSelectFilterNum()];
//        MLDataSet set = new BasicMLDataSet();
//        for (int j = 0; j < 100; j++) {
//            for (int i = 0; i < modelConfig.getVarSelectFilterNum(); i++) {
//                input[i] = random.nextDouble();
//            }
//
//            double[] ideal = new double[1];
//            ideal[0] = random.nextInt(2);
//            MLDataPair pair = new BasicMLDataPair(new BasicMLData(input),
//                    new BasicMLData(ideal));
//
//            set.add(pair);
//        }
//
//        NNTrainer trainer = new NNTrainer(modelConfig, 0, false);
//        trainer.setBaseMSE(100.);
//        trainer.setDataSet(set);
//        trainer.buildNetwork();
//
//        //case 1
//        modelConfig.getVarSelect().setWrapperBy("S");
//        VariableSelector selector = new VariableSelector(modelConfig, columnConfigList);
//        selector.selectByFilter();
//        trainer.setBaseMSE(100.);
//        selector.selectByWrapper(trainer);
//
//        int selectedNum = 0;
//        for (ColumnConfig config : columnConfigList) {
//            if (config.isFinalSelect()) {
//                selectedNum++;
//            }
//        }
//        Assert.assertEquals(selectedNum, 20);
//
//        //case 2
//        modelConfig.getVarSelect().setWrapperBy("A");
//        modelConfig.getVarSelect().setWrapperNum(20);
//        selector = new VariableSelector(modelConfig, columnConfigList);
//        trainer.setBaseMSE(100.);
//        selector.selectByWrapper(trainer);
//
//        selectedNum = 0;
//        for (ColumnConfig config : columnConfigList) {
//            if (config.isFinalSelect()) {
//                selectedNum++;
//            }
//        }
//        Assert.assertEquals(selectedNum, 20);
//
//        //case 3
//        modelConfig.getVarSelect().setWrapperBy("R");
//        modelConfig.getVarSelect().setWrapperNum(20);
//        selector = new VariableSelector(modelConfig, columnConfigList);
//        trainer.setBaseMSE(100.);
//        selector.selectByWrapper(trainer);
//        selectedNum = 0;
//        for (ColumnConfig config : columnConfigList) {
//            if (config.isFinalSelect()) {
//                selectedNum++;
//            }
//        }
//        Assert.assertEquals(selectedNum, 20);
//    }

}
