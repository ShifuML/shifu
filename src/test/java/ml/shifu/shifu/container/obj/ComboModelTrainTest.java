/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.container.obj;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.util.JSONUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Created by zhanhu on 11/18/16.
 */
public class ComboModelTrainTest {

    @Test
    public void testSerDeser() throws IOException {
        ComboModelTrain inst = new ComboModelTrain();

        inst.setUidColumnName("txnId");
        List<VarTrainConf> varTrainConfList = new ArrayList<VarTrainConf>();
        varTrainConfList.add(createVarTrainConf(ModelTrainConf.ALGORITHM.NN));
        varTrainConfList.add(createVarTrainConf(ModelTrainConf.ALGORITHM.GBT));

        inst.setVarTrainConfList(varTrainConfList);
        inst.setFusionModelTrainConf(createModelTrainConf(ModelTrainConf.ALGORITHM.LR));

        JSONUtils.writeValue(new File("src/test/resources/example/combotrain/ComboTrain.json"), inst);

        ComboModelTrain anotherInst =
                JSONUtils.readValue(new File("src/test/resources/example/combotrain/ComboTrain.json"),
                        ComboModelTrain.class);
        Assert.assertEquals(inst, anotherInst);
    }

    private VarTrainConf createVarTrainConf(ModelTrainConf.ALGORITHM alg) {
        VarTrainConf varTrainConf = new VarTrainConf();
        varTrainConf.setVariables(new ArrayList<String>());
        varTrainConf.setModelTrainConf(createModelTrainConf(alg));
        return varTrainConf;
    }

    private ModelTrainConf createModelTrainConf(ModelTrainConf.ALGORITHM alg) {
        ModelTrainConf trainConf = new ModelTrainConf();

        trainConf.setAlgorithm(alg.name());
        trainConf.setEpochsPerIteration(1);
        trainConf.setParams(ModelTrainConf.createParamsByAlg(alg, trainConf));
        trainConf.setNumTrainEpochs(100);
        if(ModelTrainConf.ALGORITHM.NN.equals(alg)) {
            trainConf.setNumTrainEpochs(200);
        } else if(ModelTrainConf.ALGORITHM.SVM.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        } else if(ModelTrainConf.ALGORITHM.RF.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        }
        trainConf.setBaggingWithReplacement(true);

        return trainConf;
    }
}
