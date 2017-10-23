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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.shifu.actor.worker.DataLoadWorker;
import ml.shifu.shifu.actor.worker.DataPrepareWorker;
import ml.shifu.shifu.actor.worker.PostTrainWorker;
import ml.shifu.shifu.actor.worker.RunModelWorker;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.message.ExceptionMessage;
import ml.shifu.shifu.message.ScanEvalDataMessage;
import ml.shifu.shifu.message.StatsResultMessage;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.JSONUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.actor.UntypedActorFactory;
import akka.routing.RoundRobinRouter;

/**
 * PostTrainActor class do the post train for models. Post-Train is used the
 * training data to evaluate the models: calculate the score, and the binning range
 * for each column. Finally, post-train will calculate the average score for each column binning
 */
public class PostTrainActor extends AbstractActor {

    private static Logger log = LoggerFactory.getLogger(PostTrainActor.class);

    private ActorRef dataLoadRef;
    private ActorRef modelRunRef;
    private ActorRef dataPrepRef;
    private Map<Integer, ActorRef> columnNumToActorMap;

    private int resultCnt;
    private int expectedResultCnt;

    public PostTrainActor(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList, final AkkaExecStatus akkaStatus) {
        super(modelConfig, columnConfigList, akkaStatus);

        final ActorRef parentActorRef = getSelf();

        resultCnt = 0;

        // actors to calculate the average score for each column binning
        columnNumToActorMap = new HashMap<Integer, ActorRef>();
        for (ColumnConfig config : columnConfigList) {
            if (config.isCandidate(super.hasCandidates) && config.isFinalSelect()) {
                expectedResultCnt++;
                ActorRef actor = getContext().actorOf(new Props(new UntypedActorFactory() {
                    private static final long serialVersionUID = -4461572845675918681L;

                    public UntypedActor create() {
                        return new PostTrainWorker(modelConfig, columnConfigList, parentActorRef, parentActorRef);
                    }
                }), "Column" + config.getColumnNum() + "PostTrainActor");
                this.columnNumToActorMap.put(config.getColumnNum(), actor);
            }
        }

        // actors to convert score to column-oriented
        dataPrepRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -5719806635080547488L;

            public UntypedActor create() throws IOException {
                return new DataPrepareWorker(modelConfig, columnConfigList, parentActorRef, columnNumToActorMap);
            }
        }).withRouter(new RoundRobinRouter(this.modelConfig.getBaggingNum())), "DataPrepWorker");

        // actors to calculate score
        modelRunRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -5719806635080547488L;

            public UntypedActor create() throws IOException {
                return new RunModelWorker(modelConfig, columnConfigList, null, parentActorRef, dataPrepRef);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))), "modelRunWorker");

        // actors to load data
        dataLoadRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -6869659846227133318L;

            public UntypedActor create() {
                return new DataLoadWorker(modelConfig, columnConfigList, parentActorRef, modelRunRef);
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
        } else if (message instanceof StatsResultMessage) {
            StatsResultMessage statsRstMsg = (StatsResultMessage) message;
            ColumnConfig columnConfig = statsRstMsg.getColumnConfig();
            columnConfigList.set(columnConfig.getColumnNum(), columnConfig);

            resultCnt++;
            log.debug("Received " + resultCnt + " messages, expected message count is:" + expectedResultCnt);
            if (resultCnt == expectedResultCnt) {
                log.info("Finished post-train.");
                PathFinder pathFinder = new PathFinder(modelConfig);
                JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath()), columnConfigList);
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
