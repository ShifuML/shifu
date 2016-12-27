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
package ml.shifu.shifu.actor;

import akka.actor.*;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.alg.LogisticRegressionTrainer;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


/**
 * TrainModelActorTest class
 */
public class TrainModelActorTest {

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private ActorSystem actorSystem;

    @BeforeClass
    public void setUp() throws IOException {
        modelConfig = CommonUtils.loadModelConfig("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json", SourceType.LOCAL);
        columnConfigList = CommonUtils.loadColumnConfigList("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json", SourceType.LOCAL);
    }

    @Test
    public void testActor() throws IOException, InterruptedException {
        File tmpDir = new File("./tmp");
        FileUtils.forceMkdir(tmpDir);

        // create normalize data
        actorSystem = ActorSystem.create("shifuActorSystem");
        ActorRef normalizeRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 6777309320338075269L;

            public UntypedActor create() throws IOException {
                return new NormalizeDataActor(modelConfig, columnConfigList, new AkkaExecStatus(true));
            }
        }), "normalize-calculator");


        List<Scanner> scanners = ShifuFileUtils.getDataScanners("src/test/resources/example/cancer-judgement/DataStore/DataSet1", SourceType.LOCAL);
        normalizeRef.tell(new AkkaActorInputMessage(scanners), normalizeRef);

        while (!normalizeRef.isTerminated()) {
            Thread.sleep(5000);
        }

        File outputFile = new File("./tmp/NormalizedData");
        Assert.assertTrue(outputFile.exists());

        // start to run trainer
        actorSystem = ActorSystem.create("shifuActorSystem");
        File models = new File("./models");
        FileUtils.forceMkdir(models);

        final List<AbstractTrainer> trainers = new ArrayList<AbstractTrainer>();
        for (int i = 0; i < 5; i++) {
            AbstractTrainer trainer;
            if (modelConfig.getAlgorithm().equalsIgnoreCase("NN")) {
                trainer = new NNTrainer(this.modelConfig, i, false);
            } else if (modelConfig.getAlgorithm().equalsIgnoreCase("SVM")) {
                trainer = new SVMTrainer(this.modelConfig, i, false);
            } else if (modelConfig.getAlgorithm().equalsIgnoreCase("LR")) {
                trainer = new LogisticRegressionTrainer(this.modelConfig, i, false);
            } else {
                throw new RuntimeException("unsupport algorithm");
            }
            trainers.add(trainer);
        }

        // train model
        ActorRef modelTrainRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 6777309320338075269L;

            public UntypedActor create() throws IOException {
                return new TrainModelActor(modelConfig, columnConfigList, new AkkaExecStatus(true), trainers);
            }
        }), "trainer");

        scanners = ShifuFileUtils.getDataScanners("./tmp/NormalizedData", SourceType.LOCAL);
        modelTrainRef.tell(new AkkaActorInputMessage(scanners), modelTrainRef);

        while (!modelTrainRef.isTerminated()) {
            Thread.sleep(5000);
        }

        for (Scanner scanner : scanners) {
            scanner.close();
        }

        File model0 = new File("./models/model0.nn");
        File model1 = new File("./models/model0.nn");
        File model2 = new File("./models/model0.nn");
        File model3 = new File("./models/model0.nn");
        File model4 = new File("./models/model0.nn");

        Assert.assertTrue(model0.exists());
        Assert.assertTrue(model1.exists());
        Assert.assertTrue(model2.exists());
        Assert.assertTrue(model3.exists());
        Assert.assertTrue(model4.exists());

        File modelsTemp = new File("./modelsTmp");

        FileUtils.deleteDirectory(modelsTemp);
        FileUtils.deleteDirectory(models);
        FileUtils.deleteDirectory(tmpDir);
    }

}
