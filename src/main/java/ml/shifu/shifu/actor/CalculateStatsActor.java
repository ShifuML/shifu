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

import ml.shifu.shifu.actor.worker.DataFilterWorker;
import ml.shifu.shifu.actor.worker.DataLoadWorker;
import ml.shifu.shifu.actor.worker.DataPrepareWorker;
import ml.shifu.shifu.actor.worker.StatsCalculateWorker;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.message.ExceptionMessage;
import ml.shifu.shifu.message.ScanStatsRawDataMessage;
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
 * CalculateStatsActor class is used to calculate stats for each column.
 * Notice: Target Column or Meta data column won't calculate stats
 */
public class CalculateStatsActor extends AbstractActor {

    private static Logger log = LoggerFactory.getLogger(CalculateStatsActor.class);

    private ActorRef dataLoadRef;
    private ActorRef dataFilterRef;
    private ActorRef dataPrepRef;
    private Map<Integer, ActorRef> columnNumToActorMap;
    private int resultCnt;

    public CalculateStatsActor(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList, final AkkaExecStatus akkaStatus) {
        super(modelConfig, columnConfigList, akkaStatus);
        log.info("Creating Master Actor ...");
        log.info("AvailableProcessors: " + Runtime.getRuntime().availableProcessors());

        resultCnt = 0;
        final ActorRef parentActorRef = getSelf();

        // actors for stats calculation
        columnNumToActorMap = new HashMap<Integer, ActorRef>();
        for (ColumnConfig config : columnConfigList) {
            if (config.isCandidate(super.hasCandidates)) {
                ActorRef actor = getContext().actorOf(new Props(new UntypedActorFactory() {
                    private static final long serialVersionUID = -6498732060654560116L;

                    public UntypedActor create() {
                        return new StatsCalculateWorker(modelConfig, columnConfigList, parentActorRef, parentActorRef);
                    }
                }), "Column" + config.getColumnNum() + "StatsActor");

                this.columnNumToActorMap.put(config.getColumnNum(), actor);
            }
        }

        // actors that convert each row data into column-oriented data
        dataPrepRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -5719806635080547488L;

            public UntypedActor create() throws IOException {
                return new DataPrepareWorker(modelConfig, columnConfigList, parentActorRef, columnNumToActorMap);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))), "DataPrepWorker");

        // actors to filter data
        dataFilterRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -1653565847681119971L;

            public UntypedActor create() throws IOException {
                return new DataFilterWorker(modelConfig, columnConfigList, parentActorRef, dataPrepRef);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))), "DataFilterWorker");

        // actors to load data
        dataLoadRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -6869659846227133318L;

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

            for (Scanner scanner : scanners) {
                dataLoadRef.tell(
                        new ScanStatsRawDataMessage(scanners.size(), scanner), getSelf());
            }
        } else if (message instanceof StatsResultMessage) {
            StatsResultMessage statsRstMsg = (StatsResultMessage) message;
            ColumnConfig columnConfig = statsRstMsg.getColumnConfig();
            columnConfigList.set(columnConfig.getColumnNum(), columnConfig);

            resultCnt++;
            if (resultCnt == columnNumToActorMap.size()) {
                log.info("Received " + resultCnt + " messages. Finished Calculating Stats.");
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
