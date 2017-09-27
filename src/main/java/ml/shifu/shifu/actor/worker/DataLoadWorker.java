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
package ml.shifu.shifu.actor.worker;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.message.NormPartRawDataMessage;
import ml.shifu.shifu.message.RunModelDataMessage;
import ml.shifu.shifu.message.ScanEvalDataMessage;
import ml.shifu.shifu.message.ScanNormInputDataMessage;
import ml.shifu.shifu.message.ScanStatsRawDataMessage;
import ml.shifu.shifu.message.ScanTrainDataMessage;
import ml.shifu.shifu.message.StatsPartRawDataMessage;
import ml.shifu.shifu.message.TrainPartDataMessage;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;

import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;

import com.google.common.base.Splitter;

/**
 * DataLoadWorker class is used to load data from all kinds of source.
 * Its input is data scanner. The output are usually List.
 */
public class DataLoadWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(DataLoadWorker.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    private static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR);

    /**
     * Basic input node count for NN model
     */
    private int inputNodeCount;

    /**
     * {@link #candidateCount} is used to check if no variable is selected. If {@link #inputNodeCount} equals
     * {@link #candidateCount}, which means no column is selected or all columns are selected.
     */
    private int candidateCount;

    public DataLoadWorker(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.modelConfig.getNormalizeType(), this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.candidateCount = inputOutputIndex[2];
    }

    /*
     * (non-Javadoc)
     * 
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) {
        if(message instanceof ScanStatsRawDataMessage) {
            log.info("DataLoaderActor Starting ...");
            ScanStatsRawDataMessage msg = (ScanStatsRawDataMessage) message;
            Scanner scanner = msg.getScanner();
            int totalMsgCnt = msg.getTotalMsgCnt();

            List<String> rawDataList = readDataIntoList(scanner);

            log.info("DataLoaderActor Finished: Loaded " + rawDataList.size() + " Records.");
            nextActorRef.tell(new StatsPartRawDataMessage(totalMsgCnt, rawDataList), getSelf());
        } else if(message instanceof ScanNormInputDataMessage) {
            log.info("DataLoaderActor Starting ...");
            ScanNormInputDataMessage msg = (ScanNormInputDataMessage) message;
            Scanner scanner = msg.getScanner();
            int totalMsgCnt = msg.getTotalMsgCnt();

            List<String> rawDataList = readDataIntoList(scanner);

            log.info("DataLoaderActor Finished: Loaded " + rawDataList.size() + " Records.");
            nextActorRef.tell(new NormPartRawDataMessage(totalMsgCnt, rawDataList), getSelf());
        } else if(message instanceof ScanTrainDataMessage) {
            ScanTrainDataMessage msg = (ScanTrainDataMessage) message;
            Scanner scanner = msg.getScanner();
            int totalMsgCnt = msg.getTotalMsgCnt();

            List<MLDataPair> mlDataPairList = readTrainingData(scanner, msg.isDryRun());
            log.info("DataLoaderActor Finished: Loaded " + mlDataPairList.size() + " Records for Training.");
            nextActorRef.tell(new TrainPartDataMessage(totalMsgCnt, msg.isDryRun(), mlDataPairList), getSelf());
        } else if(message instanceof ScanEvalDataMessage) {
            log.info("DataLoaderActor Starting ...");
            ScanEvalDataMessage msg = (ScanEvalDataMessage) message;
            Scanner scanner = msg.getScanner();
            int streamId = msg.getStreamId();
            int totalStreamCnt = msg.getTotalStreamCnt();

            splitDataIntoMultiMessages(streamId, totalStreamCnt, scanner,
                    Environment.getInt(Environment.RECORD_CNT_PER_MESSAGE, 100000));

            /*
             * List<String> evalDataList = readDataIntoList(scanner);
             * 
             * log.info("DataLoaderActor Finished: Loaded " + evalDataList.size() + " Records.");
             * nextActorRef.tell( new RunModelDataMessage(totalMsgCnt, evalDataList), getSelf());
             */
        } else {
            unhandled(message);
        }
    }

    private long splitDataIntoMultiMessages(int streamId, int totalStreamCnt, Scanner scanner, int recordCntPerMsg) {
        long recordCnt = 0;
        int msgId = 0;

        List<String> rawDataList = new LinkedList<String>();

        while(scanner.hasNextLine()) {
            String raw = scanner.nextLine();
            recordCnt++;
            rawDataList.add(raw);

            if(recordCnt % recordCntPerMsg == 0) {
                log.info("Read " + recordCnt + " Records.");
                nextActorRef.tell(new RunModelDataMessage(streamId, totalStreamCnt, (msgId++), false, rawDataList),
                        getSelf());
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
     * @param scanner
     *            - input partition
     * @return list of data
     */
    public List<String> readDataIntoList(Scanner scanner) {
        List<String> rawDataList = new LinkedList<String>();

        int cntTotal = 0;
        while(scanner.hasNextLine()) {
            String raw = scanner.nextLine();
            rawDataList.add(raw);

            cntTotal++;
            if(cntTotal % 100000 == 0) {
                log.info("Read " + cntTotal + " records.");
            }
        }

        log.info("Totally read " + cntTotal + " records.");
        return rawDataList;
    }

    /**
     * Read the normalized training data for model training
     * 
     * @param scanner
     *            - input partition
     * @param isDryRun
     *            - is for test running?
     * @return List of data
     */
    public List<MLDataPair> readTrainingData(Scanner scanner, boolean isDryRun) {
        List<MLDataPair> mlDataPairList = new ArrayList<MLDataPair>();

        int numSelected = 0;
        for(ColumnConfig config: columnConfigList) {
            if(config.isFinalSelect()) {
                numSelected++;
            }
        }

        int cnt = 0;
        while(scanner.hasNextLine()) {
            if((cnt++) % 100000 == 0) {
                log.info("Read " + (cnt) + " Records.");
            }

            String line = scanner.nextLine();
            if(isDryRun) {
                MLDataPair dummyPair = new BasicMLDataPair(new BasicMLData(new double[1]), new BasicMLData(
                        new double[1]));
                mlDataPairList.add(dummyPair);
                continue;
            }

            // the normalized training data is separated by | by default
            double[] inputs = new double[numSelected];
            double[] ideal = new double[1];
            double significance = 0.0d;
            int index = 0, inputsIndex = 0, outputIndex = 0;
            for(String input: DEFAULT_SPLITTER.split(line.trim())) {
                double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
                if(index == this.columnConfigList.size()) {
                    significance = NumberFormatUtils
                            .getDouble(input.trim(), CommonConstants.DEFAULT_SIGNIFICANCE_VALUE);
                    break;
                } else {
                    ColumnConfig columnConfig = this.columnConfigList.get(index);

                    if(columnConfig != null && columnConfig.isTarget()) {
                        ideal[outputIndex++] = doubleValue;
                    } else {
                        if(this.inputNodeCount == this.candidateCount) {
                            // all variables are not set final-select
                            if(CommonUtils.isGoodCandidate(columnConfig, super.hasCandidates)) {
                                inputs[inputsIndex++] = doubleValue;
                            }
                        } else {
                            // final select some variables
                            if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                    && columnConfig.isFinalSelect()) {
                                inputs[inputsIndex++] = doubleValue;
                            }
                        }
                    }
                }
                index++;
            }

            MLDataPair pair = new BasicMLDataPair(new BasicMLData(inputs), new BasicMLData(ideal));
            pair.setSignificance(significance);

            mlDataPairList.add(pair);
        }

        return mlDataPairList;
    }
}
