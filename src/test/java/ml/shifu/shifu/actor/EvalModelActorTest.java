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

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.actor.UntypedActorFactory;


/**
 * EvalModelActorTest class
 */
public class EvalModelActorTest {

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private EvalConfig evalConfig;

    private ActorSystem actorSystem;

    @BeforeClass
    public void setUp() throws IOException {
        modelConfig = CommonUtils.loadModelConfig("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json", SourceType.LOCAL);
        columnConfigList = CommonUtils.loadColumnConfigList("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json", SourceType.LOCAL);
        evalConfig = modelConfig.getEvalConfigByName("EvalA");
        actorSystem = ActorSystem.create("shifuActorSystem");
    }

//    @Test
    public void testActor() throws IOException, InterruptedException {
        Environment.setProperty(Environment.SHIFU_HOME, ".");

        File tmpModels = new File("models");
        File tmpCommon = new File("common");

        File models = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");
        FileUtils.copyDirectory(models, tmpModels);

        File tmpEvalA = new File("evals");
        FileUtils.forceMkdir(tmpEvalA);
        
        ActorRef modelEvalRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -1437127862571741369L;

            public UntypedActor create() {
                return new EvalModelActor(modelConfig, columnConfigList, new AkkaExecStatus(true), evalConfig);
            }
        }), "model-evaluator");

        List<Scanner> scanners = ShifuFileUtils.getDataScanners("src/test/resources/example/cancer-judgement/DataStore/EvalSet1", SourceType.LOCAL);
        modelEvalRef.tell(new AkkaActorInputMessage(scanners), modelEvalRef);

        while (!modelEvalRef.isTerminated()) {
            Thread.sleep(5000);
        }


        File outputFile = new File("evals/EvalA/EvalScore");
        Assert.assertTrue(outputFile.exists());

        FileUtils.deleteDirectory(tmpModels);
        FileUtils.deleteDirectory(tmpCommon);
        FileUtils.deleteDirectory(tmpEvalA);
    }

}
