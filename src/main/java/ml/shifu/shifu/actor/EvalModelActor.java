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

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.actor.UntypedActorFactory;
import akka.routing.RoundRobinRouter;
import ml.shifu.shifu.actor.worker.DataFilterWorker;
import ml.shifu.shifu.actor.worker.DataLoadWorker;
import ml.shifu.shifu.actor.worker.RunModelWorker;
import ml.shifu.shifu.actor.worker.ScoreModelWorker;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.message.EvalResultMessage;
import ml.shifu.shifu.message.ExceptionMessage;
import ml.shifu.shifu.message.ScanEvalDataMessage;
import ml.shifu.shifu.util.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;


/**
 * EvalModelActor class to evaluate the performance of generated models.
 * If there is Target Column in the evaluation data, the actor will not only calculate
 * the score for each record, but also it will generate performance matrix.
 * If there is no Target Column, only score will be calculated.
 */
public class EvalModelActor extends AbstractActor {

    private static Logger log = LoggerFactory.getLogger(EvalModelActor.class);

    private ActorRef dataLoadRef;
    private ActorRef dataFilterRef;
    private ActorRef modelRunRef;
    private ActorRef scoreGenRef;

    private int resultCnt;

    public EvalModelActor(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList, final AkkaExecStatus akkaStatus, final EvalConfig evalConfig) {
        super(modelConfig, columnConfigList, akkaStatus);
        log.info("Creating Master Actor ...");
        log.info("AvailableProcessors: " + Runtime.getRuntime().availableProcessors());

        final ActorRef parentActorRef = getSelf();
        resultCnt = 0;

        // actors to record the score and generate score file header
        scoreGenRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 52071053911068732L;

            public UntypedActor create() throws IOException {
                return new ScoreModelWorker(modelConfig, columnConfigList, parentActorRef, parentActorRef, evalConfig);
            }
        }), "ScoreGenWorker");

        // actors to run the models
        modelRunRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 1765987019989131341L;

            public UntypedActor create() throws IOException {
                return new RunModelWorker(modelConfig, columnConfigList, evalConfig, parentActorRef, scoreGenRef);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))), "ModelRunWorker");

        // actors to filter data
        dataFilterRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 867848826648611314L;

            public UntypedActor create() throws IOException {
                return new DataFilterWorker(modelConfig, columnConfigList, parentActorRef, modelRunRef, evalConfig);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))), "DataFilterWorker");

        // actors to load data
        dataLoadRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 3128128348796746167L;

            public UntypedActor create() {
                return new DataLoadWorker(modelConfig, columnConfigList, parentActorRef, dataFilterRef);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))), "DataLoaderWorker");
    }

    /* (non-Javadoc)
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof AkkaActorInputMessage) {
            resultCnt = 0;

            AkkaActorInputMessage msg = (AkkaActorInputMessage) message;
            List<Scanner> scanners = msg.getScanners();

            log.debug("Num of Scanners: " + scanners.size());
            int streamId = 0;

            for (Scanner scanner : scanners) {
                dataLoadRef.tell(new ScanEvalDataMessage(streamId++, scanners.size(), scanner), getSelf());
            }
        } else if (message instanceof EvalResultMessage) {
            EvalResultMessage msg = (EvalResultMessage) message;
            resultCnt++;
            if (resultCnt == msg.getTotalMsgCnt()) {
                getContext().system().shutdown();
            }
        } else if (message instanceof ExceptionMessage) {
            // since some children actors meet some exception, shutdown the system
            ExceptionMessage msg = (ExceptionMessage) message;
            getContext().system().shutdown();

            // and wrapper the exception into Return status
            addExceptionIntoCondition(msg.getException());
        } else {
            unhandled(message);
        }
    }
}