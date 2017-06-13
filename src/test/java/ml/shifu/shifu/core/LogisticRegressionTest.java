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

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import org.apache.commons.io.FileUtils;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Random;


public class LogisticRegressionTest {

    LogisticRegressionTrainer trainer;
    ModelConfig config;

    MLDataSet trainSet;

    Random random;

    @BeforeClass
    public void setUp() throws IOException {
        random = new Random();

        config = ModelConfig.createInitModelConfig("test", ALGORITHM.LR, "test", false);

        config.getVarSelect().setFilterNum(5);
        config.getTrain().setAlgorithm("LR");
        //config.
        config.getTrain().setNumTrainEpochs(100);
        config.getTrain().setParams(new HashMap<String, Object>());
        config.getTrain().getParams().put("LearningRate", 0.1);

        trainer = new LogisticRegressionTrainer(config, 0, false);

        trainSet = new BasicMLDataSet();

        for (int i = 0; i < 1000; i++) {
            double[] input = new double[5];
            double[] ideal = new double[1];

            for (int j = 0; j < 5; j++) {
                input[j] = random.nextDouble();
            }

            ideal[0] = random.nextInt(2);

            MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
            trainSet.add(pair);
        }

        trainer.setDataSet(trainSet);
        trainer.setValidSet(trainSet);
    }

    @Test
    public void test() throws IOException {
        trainer.train();
        File file = new File("models/model0.lr");
        Assert.assertTrue(file.exists());

        Assert.assertTrue(trainer.getClassifier().getStructure().getFlat().getWeights().length == 6);
    }

    @AfterClass
    public void tearDown() throws IOException {
        FileUtils.deleteDirectory(new File("./models/"));
        FileUtils.deleteDirectory(new File("./modelsTmp/"));
        FileUtils.deleteQuietly(new File("ModelConfig.json"));

        FileUtils.deleteDirectory(new File("test"));
    }

}
