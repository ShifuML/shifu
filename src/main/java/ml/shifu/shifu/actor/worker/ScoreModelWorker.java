/**
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

import akka.actor.ActorRef;
import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.EvalResultMessage;
import ml.shifu.shifu.message.RunModelResultMessage;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * ScoreModelWorker class collect all the score for evaluation data and save
 * them into file. If the evaluation data contains target column, it will also
 * calculate the performance matrix.
 */
public class ScoreModelWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(ScoreModelWorker.class);
    private EvalConfig evalConfig;

    private String[] header;
    private BufferedWriter scoreWriter;
    // private Reasoner reasoner;
    private int receivedStreamCnt;
    private Map<Integer, StreamBulletin> resultMap;

    public ScoreModelWorker(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, ActorRef parentActorRef,
            ActorRef nextActorRef, EvalConfig evalConfig) throws IOException {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        this.evalConfig = evalConfig;

        PathFinder pathFinder = new PathFinder(modelConfig);

        // make sure local directory - evals/<EvalSetName> exists
        ShifuFileUtils.createDirIfNotExists(pathFinder.getEvalSetPath(evalConfig), evalConfig.getDataSet().getSource());

        // clear output - evals/<EvalSetName>/EvalScore at first,
        // for it may be directory
        ShifuFileUtils.deleteFile(pathFinder.getEvalScorePath(evalConfig), evalConfig.getDataSet().getSource());

        // create score writer
        scoreWriter = ShifuFileUtils.getWriter(pathFinder.getEvalScorePath(evalConfig), evalConfig.getDataSet()
                .getSource());

        // load the header for evaluation data
        header = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), evalConfig.getDataSet()
                .getHeaderDelimiter(), evalConfig.getDataSet().getSource());

        writeScoreHeader();

        receivedStreamCnt = 0;
        resultMap = new HashMap<Integer, StreamBulletin>();
    }

    /*
     * (non-Javadoc)
     * 
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) throws IOException {
        if(message instanceof RunModelResultMessage) {
            log.debug("Received model score data for evaluation");
            RunModelResultMessage msg = (RunModelResultMessage) message;
            if(!resultMap.containsKey(msg.getStreamId())) {
                receivedStreamCnt++;
                resultMap.put(msg.getStreamId(), new StreamBulletin(msg.getStreamId()));
            }
            resultMap.get(msg.getStreamId()).receiveMsge(msg.getMsgId(), msg.isLastMsg());

            List<CaseScoreResult> caseScoreResultList = msg.getScoreResultList();

            StringBuilder buf = new StringBuilder();
            for(CaseScoreResult csResult: caseScoreResultList) {
                buf.setLength(0);

                Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(csResult.getInputData(), evalConfig
                        .getDataSet().getDataDelimiter(), header);

                // get the tag
                String tag = rawDataMap.get(modelConfig.getTargetColumnName(evalConfig));
                buf.append(StringUtils.trimToEmpty(tag));

                // append weight column value
                if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())) {
                    String metric = rawDataMap.get(evalConfig.getDataSet().getWeightColumnName());
                    buf.append("|" + StringUtils.trimToEmpty(metric));
                } else {
                    buf.append("|" + "1.0");
                }

                buf.append("|" + csResult.getAvgScore());
                buf.append("|" + csResult.getMaxScore());
                buf.append("|" + csResult.getMinScore());
                buf.append("|" + csResult.getMedianScore());

                // score
                for(Integer score: csResult.getScores()) {
                    buf.append("|" + score);
                }

                // append meta data
                List<String> metaColumns = evalConfig.getScoreMetaColumns(modelConfig);
                if(CollectionUtils.isNotEmpty(metaColumns)) {
                    for(String columnName: metaColumns) {
                        String value = rawDataMap.get(columnName);
                        buf.append("|" + StringUtils.trimToEmpty(value));
                    }
                }

                scoreWriter.write(buf.toString() + "\n");
            }

            if(receivedStreamCnt == msg.getTotalStreamCnt() && hasAllMessageResult(resultMap)) {
                log.info("Finish running scoring, the score file - {} is stored in {}.", new PathFinder(modelConfig)
                        .getEvalScorePath(evalConfig).toString(), evalConfig.getDataSet().getSource().name());
                scoreWriter.close();

                // only one message will be sent
                nextActorRef.tell(new EvalResultMessage(1), this.getSelf());
            }
        } else {
            unhandled(message);
        }
    }

    /**
     * @param resultMap2
     * @return
     */
    private boolean hasAllMessageResult(Map<Integer, StreamBulletin> resultMsgMap) {
        Iterator<Entry<Integer, StreamBulletin>> iterator = resultMsgMap.entrySet().iterator();
        while(iterator.hasNext()) {
            Entry<Integer, StreamBulletin> entry = iterator.next();
            if(!entry.getValue().isMessageEnd()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Write the file header for score file
     * 
     * @throws IOException
     */
    private void writeScoreHeader() throws IOException {
        StringBuilder buf = new StringBuilder();
        buf.append(modelConfig.getTargetColumnName(evalConfig) == null ? "tag" : modelConfig
                .getTargetColumnName(evalConfig));

        buf.append("|"
                + (StringUtils.isBlank(evalConfig.getDataSet().getWeightColumnName()) ? "weight" : evalConfig
                        .getDataSet().getWeightColumnName()));

        buf.append("|mean|max|min|median");

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, columnConfigList, evalConfig, SourceType.LOCAL);
        for(int i = 0; i < models.size(); i++) {
            buf.append("|model" + i);
        }

        // append meta data
        List<String> metaColumns = evalConfig.getScoreMetaColumns(modelConfig);
        if(CollectionUtils.isNotEmpty(metaColumns)) {
            for(String columnName: metaColumns) {
                buf.append("|" + columnName);
            }
        }

        scoreWriter.write(buf.toString() + "\n");
    }

    public static class StreamBulletin {
        private int streamId;
        private long targetSum;
        private long totalSum;
        private boolean hasLastMsg;

        public StreamBulletin(int streamId) {
            this.streamId = streamId;
            this.totalSum = 0;
            this.targetSum = 0;
            this.hasLastMsg = false;
        }

        public void receiveMsge(int msgId, boolean isLastMsg) {
            if(isLastMsg) {
                hasLastMsg = true;
                targetSum = msgId * (msgId + 1) / 2;
            }

            totalSum = totalSum + msgId;
        }

        public int getParallId() {
            return this.streamId;
        }

        public boolean isMessageEnd() {
            return hasLastMsg && (totalSum == targetSum);
        }
    }
}
