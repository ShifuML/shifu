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

import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

/**
 * Created by zhanhu on 11/18/16.
 */
public class ComboModelTrainTest {

    @Test
    public void testSerDeser() throws IOException {
        ComboModelTrain inst = new ComboModelTrain();

        List<SubTrainConf> varTrainConfList = new ArrayList<SubTrainConf>();
        varTrainConfList.add(createSubTrainConf(ModelTrainConf.ALGORITHM.NN));
        varTrainConfList.add(createSubTrainConf(ModelTrainConf.ALGORITHM.GBT));

        inst.setSubTrainConfList(varTrainConfList);

        JSONUtils.writeValue(new File("src/test/resources/example/ComboTrain.json"), inst);

        ComboModelTrain anotherInst = JSONUtils.readValue(new File(
                "src/test/resources/example/ComboTrain.json"), ComboModelTrain.class);
        Assert.assertEquals(inst.getSubTrainConfList().size(), anotherInst.getSubTrainConfList().size());
    }

    private SubTrainConf createSubTrainConf(ModelTrainConf.ALGORITHM alg) {
        SubTrainConf subTrainConf = new SubTrainConf();
        subTrainConf.setModelStatsConf(createModelStatsConf(alg));
        subTrainConf.setModelNormalizeConf(createModelNormalizeConf(alg));
        subTrainConf.setModelVarSelectConf(createModelVarSelectConf(alg));
        subTrainConf.setModelTrainConf(createModelTrainConf(alg));
        return subTrainConf;
    }

    private ModelStatsConf createModelStatsConf(ModelTrainConf.ALGORITHM alg) {
        ModelStatsConf statsConf = new ModelStatsConf();
        if(ModelTrainConf.ALGORITHM.NN.equals(alg) || ModelTrainConf.ALGORITHM.LR.equals(alg )) {
            statsConf.setBinningAlgorithm(ModelStatsConf.BinningAlgorithm.DynamicBinning);
            statsConf.setBinningMethod(ModelStatsConf.BinningMethod.EqualTotal);
            statsConf.setMaxNumBin(20);
        } else if(ModelTrainConf.ALGORITHM.RF.equals(alg) || ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            statsConf.setBinningAlgorithm(ModelStatsConf.BinningAlgorithm.SPDTI);
            statsConf.setBinningMethod(ModelStatsConf.BinningMethod.EqualPositive);
            statsConf.setMaxNumBin(20);
        }
        return statsConf;
    }

    private ModelNormalizeConf createModelNormalizeConf(ModelTrainConf.ALGORITHM alg) {
        ModelNormalizeConf normalizeConf = new ModelNormalizeConf();
        normalizeConf.setNormType(ModelNormalizeConf.NormType.WOE);
        normalizeConf.setSampleNegOnly(false);
        normalizeConf.setSampleRate(1.0);
        return normalizeConf;
    }

    private ModelVarSelectConf createModelVarSelectConf(ModelTrainConf.ALGORITHM alg) {
        ModelVarSelectConf varSelectConf = new ModelVarSelectConf();
        varSelectConf.setFilterNum(20);
        if(ModelTrainConf.ALGORITHM.NN.equals(alg) || ModelTrainConf.ALGORITHM.LR.equals(alg )) {
            varSelectConf.setFilterBy("SE");
        } else if(ModelTrainConf.ALGORITHM.RF.equals(alg) || ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            varSelectConf.setFilterBy("FI");
        }
        return varSelectConf;
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

    @AfterClass
    public void cleanup() {
        FileUtils.deleteQuietly(new File("src/test/resources/example/ComboTrain.json"));
    }

}
