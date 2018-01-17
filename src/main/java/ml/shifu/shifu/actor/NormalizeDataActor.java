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

import java.io.BufferedWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Scanner;

import ml.shifu.shifu.actor.worker.DataFilterWorker;
import ml.shifu.shifu.actor.worker.DataLoadWorker;
import ml.shifu.shifu.actor.worker.DataNormalizeWorker;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import ml.shifu.shifu.message.ExceptionMessage;
import ml.shifu.shifu.message.NormResultDataMessage;
import ml.shifu.shifu.message.ScanNormInputDataMessage;
import ml.shifu.shifu.util.Environment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.actor.UntypedActorFactory;
import akka.routing.RoundRobinRouter;

/**
 * NormalizeDataActor class normalize the the training data.
 * Not all training data will be normalized. Actually there is an option - `baggingSampleRate`
 * in @ModelConfig, it will control how many percentage will be normalized.
 * <p>
 * The raw data which is normalized, is also will be stored.
 */
public class NormalizeDataActor extends AbstractActor {

    private static Logger log = LoggerFactory.getLogger(NormalizeDataActor.class);

    private ActorRef dataLoadRef;
    private ActorRef dataFilterRef;
    private ActorRef dataNormalizeRef;

    private BufferedWriter normDataWriter;
    private BufferedWriter selectDataWriter;

    private DecimalFormat df;
    private int resultCnt;

    public NormalizeDataActor(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            final AkkaExecStatus akkaStatus) throws IOException {
        super(modelConfig, columnConfigList, akkaStatus);
        log.info("Creating Master Actor ...");
        log.info("AvailableProcessors: " + Runtime.getRuntime().availableProcessors());

        resultCnt = 0;
        final ActorRef parentActorRef = getSelf();

        // actors to normalize data
        dataNormalizeRef = this.getContext().actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -2417228112206743801L;

            public UntypedActor create() {
                return new DataNormalizeWorker(modelConfig, columnConfigList, parentActorRef, parentActorRef);
            }
        }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))),
                "DataNormalizeWorker");

        // actors to filter data
        dataFilterRef = this.getContext()
                .actorOf(new Props(new UntypedActorFactory() {
                    private static final long serialVersionUID = 7122505775141026832L;

                    public UntypedActor create() throws IOException {
                        return new DataFilterWorker(modelConfig, columnConfigList, parentActorRef, dataNormalizeRef);
                    }
                }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))),
                        "DataFilterWorker");

        // actors to load data
        dataLoadRef = this.getContext()
                .actorOf(new Props(new UntypedActorFactory() {
                    private static final long serialVersionUID = -7499072868479157207L;

                    public UntypedActor create() {
                        return new DataLoadWorker(modelConfig, columnConfigList, parentActorRef, dataFilterRef);
                    }
                }).withRouter(new RoundRobinRouter(Environment.getInt(Environment.LOCAL_NUM_PARALLEL, 16))),
                        "DataLoaderWorker");

        PathFinder pathFinder = new PathFinder(modelConfig);
        SourceType sourceType = modelConfig.getDataSet().getSource();
        normDataWriter = ShifuFileUtils.getWriter(pathFinder.getNormalizedDataPath(sourceType), sourceType);
        selectDataWriter = ShifuFileUtils.getWriter(pathFinder.getSelectedRawDataPath(sourceType), sourceType);

        df = new DecimalFormat("#.######");
    }

    /*
     * (non-Javadoc)
     * 
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void onReceive(Object message) throws Exception {
        if(message instanceof AkkaActorInputMessage) {
            resultCnt = 0;

            AkkaActorInputMessage msg = (AkkaActorInputMessage) message;
            List<Scanner> scanners = msg.getScanners();

            for(Scanner scanner: scanners) {
                dataLoadRef.tell(new ScanNormInputDataMessage(scanners.size(), scanner), getSelf());
            }
        } else if(message instanceof NormResultDataMessage) {
            NormResultDataMessage msg = (NormResultDataMessage) message;
            int targetMsgCnt = msg.getTargetMsgCnt();

            writeDataIntoFile(msg.getNormalizedDataList());
            writeSelectDataIntoFile(msg.getSelectDataList());

            resultCnt++;
            if(resultCnt == targetMsgCnt) {
                log.info("Received " + resultCnt + " messages. Finished normalizing train data.");
                normDataWriter.close();
                selectDataWriter.close();
                getContext().system().shutdown();
            }
        } else if(message instanceof ExceptionMessage) {
            // since some children actors meet some exception, shutdown the system
            ExceptionMessage msg = (ExceptionMessage) message;
            getContext().system().shutdown();

            // and wrapper the exception into Return status
            addExceptionIntoCondition(msg.getException());
        } else {
            unhandled(message);
        }
    }

    /**
     * Write the data which is selected to normalize, into file - tmp/SelectedRawData
     * 
     * @param selectDataList
     *            - the raw selected data
     * @throws IOException
     *             Exception when writing file
     */
    private void writeSelectDataIntoFile(List<String> selectDataList) throws IOException {
        for(String rawInput: selectDataList) {
            selectDataWriter.append(rawInput + "\n");
        }
    }

    /**
     * Write the normalized data into file - tmp/NormalizedData
     * 
     * @param normalizedDataList
     *            - the normalized data
     * @throws IOException
     *             Exception when writing file
     */
    private void writeDataIntoFile(List<List<Double>> normalizedDataList) throws IOException {
        for(List<Double> normData: normalizedDataList) {
            for(Double data: normData) {
                if(data == null) {
                    normDataWriter.append("|");
                } else {
                    normDataWriter.append(df.format(data) + "|");
                }
            }

            normDataWriter.append("\n");
        }
    }

}
