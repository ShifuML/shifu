/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.core.actor.worker;

import akka.actor.ActorRef;
import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.container.obj.ModelConfig;
import ml.shifu.core.message.*;
import ml.shifu.core.util.Environment;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;


/**
 * DataLoadWorker class is used to load data from all kinds of source.
 * Its input is data scanner. The output are usually List<String>
 */
public class DataLoadWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory
            .getLogger(DataLoadWorker.class);

    public DataLoadWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
    }

    /*
     * (non-Javadoc)
     *
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) {
        if (message instanceof ScanStatsRawDataMessage) {
            log.info("DataLoaderActor Starting ...");
            ScanStatsRawDataMessage msg = (ScanStatsRawDataMessage) message;
            Scanner scanner = msg.getScanner();
            int totalMsgCnt = msg.getTotalMsgCnt();

            List<String> rawDataList = readDataIntoList(scanner);

            log.info("DataLoaderActor Finished: Loaded " + rawDataList.size() + " Records.");
            nextActorRef.tell(new StatsPartRawDataMessage(totalMsgCnt, rawDataList), getSelf());
        } else if (message instanceof ScanNormInputDataMessage) {
            log.info("DataLoaderActor Starting ...");
            ScanNormInputDataMessage msg = (ScanNormInputDataMessage) message;
            Scanner scanner = msg.getScanner();
            int totalMsgCnt = msg.getTotalMsgCnt();

            List<String> rawDataList = readDataIntoList(scanner);

            log.info("DataLoaderActor Finished: Loaded " + rawDataList.size() + " Records.");
            nextActorRef.tell(new NormPartRawDataMessage(totalMsgCnt, rawDataList), getSelf());
        } else if (message instanceof ScanTrainDataMessage) {
            ScanTrainDataMessage msg = (ScanTrainDataMessage) message;
            Scanner scanner = msg.getScanner();
            int totalMsgCnt = msg.getTotalMsgCnt();

            List<MLDataPair> mlDataPairList = readTrainingData(scanner, msg.isDryRun());
            log.info("DataLoaderActor Finished: Loaded " + mlDataPairList.size() + " Records for Training.");
            nextActorRef.tell(new TrainPartDataMessage(totalMsgCnt, msg.isDryRun(), mlDataPairList), getSelf());
        } else if (message instanceof ScanEvalDataMessage) {
            log.info("DataLoaderActor Starting ...");
            ScanEvalDataMessage msg = (ScanEvalDataMessage) message;
            Scanner scanner = msg.getScanner();
            int streamId = msg.getStreamId();
            int totalStreamCnt = msg.getTotalStreamCnt();

            splitDataIntoMultiMessages(streamId, totalStreamCnt, scanner, Environment.getInt(Environment.RECORD_CNT_PER_MESSAGE, 100000));

			/*List<String> evalDataList = readDataIntoList(scanner);

			log.info("DataLoaderActor Finished: Loaded " + evalDataList.size() + " Records.");
			nextActorRef.tell( new RunModelDataMessage(totalMsgCnt, evalDataList), getSelf());*/
        } else {
            unhandled(message);
        }
    }

    /**
     * @param totalMsgCnt
     * @param scanner
     * @param int1
     */
    private long splitDataIntoMultiMessages(int streamId, int totalStreamCnt, Scanner scanner, int recordCntPerMsg) {
        long recordCnt = 0;
        int msgId = 0;

        List<String> rawDataList = new LinkedList<String>();

        while (scanner.hasNextLine()) {
            String raw = scanner.nextLine();
            recordCnt++;
            rawDataList.add(raw);

            if (recordCnt % recordCntPerMsg == 0) {
                log.info("Read " + recordCnt + " Records.");
                nextActorRef.tell(new RunModelDataMessage(streamId, totalStreamCnt, (msgId++), false, rawDataList), getSelf());
                rawDataList = new LinkedList<String>();
            }
        }

        log.info("Totally read " + recordCnt + " Records.");
        // anyhow, sent the last message to let next actor know - it's done
        nextActorRef.tell(new RunModelDataMessage(streamId, totalStreamCnt, (msgId++), true, rawDataList), getSelf());

        return recordCnt;
    }

    /**
     * Read data into String list
     *
     * @param scanner - input partition
     * @return List<String>
     */
    public List<String> readDataIntoList(Scanner scanner) {
        List<String> rawDataList = new LinkedList<String>();

        int cntTotal = 0;
        while (scanner.hasNextLine()) {
            String raw = scanner.nextLine();
            rawDataList.add(raw);

            cntTotal++;
            if (cntTotal % 100000 == 0) {
                log.info("Read " + cntTotal + " records.");
            }
        }

        log.info("Totally read " + cntTotal + " records.");
        return rawDataList;
    }

    /**
     * Read the normalized training data for model training
     *
     * @param scanner  - input partition
     * @param isDryRun - is for test running?
     * @return List<MLDataPair>
     */
    public List<MLDataPair> readTrainingData(Scanner scanner, boolean isDryRun) {
        List<MLDataPair> mlDataPairList = new ArrayList<MLDataPair>();

        int numSelected = 0;
        for (ColumnConfig config : columnConfigList) {
            if (config.isFinalSelect()) {
                numSelected++;
            }
        }

        int cnt = 0;
        while (scanner.hasNextLine()) {
            if ((cnt++) % 100000 == 0) {
                log.info("Read " + (cnt) + " Records.");
            }

            String line = scanner.nextLine();
            if (isDryRun) {
                MLDataPair dummyPair = new BasicMLDataPair(new BasicMLData(
                        new double[1]), new BasicMLData(new double[1]));
                mlDataPairList.add(dummyPair);
                continue;
            }

            // the normalized training data is separated by | by default
            String[] raw = line.trim().split("\\|");

            double[] input = new double[numSelected];
            double[] ideal = new double[1];

            try {
                ideal[0] = Double.parseDouble(raw[0]);
                for (int i = 0; i < numSelected; i++) {
                    input[i] = Double.valueOf(raw[i + 1]);
                }
            } catch (Exception e) { // skip invalid records
                log.debug("Can't convert data into double.", e);
                continue;
            }

            MLDataPair pair = new BasicMLDataPair(new BasicMLData(input),
                    new BasicMLData(ideal));
            pair.setSignificance(Double.valueOf(raw[raw.length - 1]));

            mlDataPairList.add(pair);
        }

        return mlDataPairList;
    }
}
