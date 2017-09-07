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
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.actor.UntypedActorFactory;

/**
 * CalculateStatsActorTest class
 */
public class CalculateStatsActorTest {

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private ActorSystem actorSystem;

    @BeforeClass
    public void setUp() throws IOException {
        modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json", SourceType.LOCAL);
        columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json", SourceType.LOCAL);
        actorSystem = ActorSystem.create("shifuActorSystem");
    }

    // @Test
    public void testActor() throws IOException, InterruptedException {
        ActorRef statsCalRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 6777309320338075269L;

            public UntypedActor create() {
                return new CalculateStatsActor(modelConfig, columnConfigList, new AkkaExecStatus(true));
            }
        }), "stats-calculator");

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(
                "src/test/resources/example/cancer-judgement/DataStore/DataSet1", SourceType.LOCAL);
        statsCalRef.tell(new AkkaActorInputMessage(scanners), statsCalRef);

        while(!statsCalRef.isTerminated()) {
            Thread.sleep(5000);
        }

        File file = new File("./ColumnConfig.json");
        Assert.assertTrue(file.exists());

    }

    @AfterClass
    public void delete() throws IOException {
        File file = new File("./ColumnConfig.json");
        if(file.exists()) {
            FileUtils.deleteQuietly(file);
        }
    }

}
