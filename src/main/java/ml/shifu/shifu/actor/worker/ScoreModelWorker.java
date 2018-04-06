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

import akka.actor.ActorRef;
import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.model.ModelSpec;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.EvalResultMessage;
import ml.shifu.shifu.message.RunModelResultMessage;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang.StringUtils;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.*;
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
    private Map<String, Integer> subModelsCnt;

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
        header = CommonUtils.getFinalHeaders(evalConfig);

        receivedStreamCnt = 0;
        resultMap = new HashMap<Integer, StreamBulletin>();

        subModelsCnt = new TreeMap<String, Integer>();
        @SuppressWarnings("deprecation")
        List<ModelSpec> subModels = CommonUtils.loadSubModels(modelConfig, this.columnConfigList, evalConfig,
                evalConfig.getDataSet().getSource(), evalConfig.getGbtConvertToProb());
        if(CollectionUtils.isNotEmpty(subModels)) {
            for(ModelSpec modelSpec: subModels) {
                System.out.println("get sub model " + modelSpec.getModelName() + "|" + modelSpec.getModels().size());
                subModelsCnt.put(modelSpec.getModelName(), modelSpec.getModels().size());
            }
        }

        writeScoreHeader();
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
                String tag = CommonUtils.trimTag(rawDataMap.get(modelConfig.getTargetColumnName(evalConfig)));
                buf.append(tag);

                // append weight column value
                if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())) {
                    String metric = rawDataMap.get(evalConfig.getDataSet().getWeightColumnName());
                    buf.append("|" + StringUtils.trimToEmpty(metric));
                } else {
                    buf.append("|" + "1.0");
                }

                if ( CollectionUtils.isNotEmpty(csResult.getScores()) ) {
                    addModelScoreData(buf, csResult);
                }

                Map<String, CaseScoreResult> subModelScores = csResult.getSubModelScores();
                if ( MapUtils.isNotEmpty(subModelScores) ) {
                    Iterator<Map.Entry<String, CaseScoreResult>> iterator = subModelScores.entrySet().iterator();
                    while(iterator.hasNext()) {
                        Map.Entry<String, CaseScoreResult> entry = iterator.next();
                        CaseScoreResult subCs = entry.getValue();
                        addModelScoreData(buf, subCs);
                    }
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

    private void addModelScoreData(StringBuilder buf, CaseScoreResult cs) {
        buf.append("|" + cs.getAvgScore());
        buf.append("|" + cs.getMaxScore());
        buf.append("|" + cs.getMinScore());
        buf.append("|" + cs.getMedianScore());

        // score
        for (Double score : cs.getScores()) {
            buf.append("|" + score);
        }
    }

    /**
     * Write the file header for score file
     * 
     * @throws IOException
     *             if any ip exception
     */
    private void writeScoreHeader() throws IOException {
        StringBuilder buf = new StringBuilder();
        buf.append(modelConfig.getTargetColumnName(evalConfig) == null ? "tag" : modelConfig
                .getTargetColumnName(evalConfig));

        buf.append("|" + (StringUtils.isBlank(evalConfig.getDataSet().getWeightColumnName())
                ? "weight" : evalConfig.getDataSet().getWeightColumnName()));

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, evalConfig, SourceType.LOCAL);
        if ( CollectionUtils.isNotEmpty(models) ) {
            addModelScoreHeader(buf, models.size(), "");
        }

        if(MapUtils.isNotEmpty(this.subModelsCnt)) {
            Iterator<Map.Entry<String, Integer>> iterator = this.subModelsCnt.entrySet().iterator();
            while(iterator.hasNext()) {
                Map.Entry<String, Integer> entry = iterator.next();
                String modelName = entry.getKey();
                Integer smCnt = entry.getValue();
                if(smCnt > 0) {
                    addModelScoreHeader(buf, smCnt, modelName);
                }
            }
        }

        // append meta data
        List<String> metaColumns = evalConfig.getAllMetaColumns(modelConfig);
        if(CollectionUtils.isNotEmpty(metaColumns)) {
            for(String columnName: metaColumns) {
                buf.append("|" + columnName);
            }
        }

        scoreWriter.write(buf.toString() + "\n");
    }

    private void addModelScoreHeader(StringBuilder buf, Integer modelCnt, String modelName) {
        buf.append("|" + addModelNameAsNS(modelName, "mean"));
        buf.append("|" + addModelNameAsNS(modelName, "max"));
        buf.append("|" + addModelNameAsNS(modelName, "min"));
        buf.append("|" + addModelNameAsNS(modelName, "median"));
        for (int i = 0; i < modelCnt; i++) {
            buf.append("|" + addModelNameAsNS(modelName, "model" + i));
        }
    }

    private String addModelNameAsNS(String modelName, String scoreName) {
        return (StringUtils.isBlank(modelName) ? scoreName : modelName + "::" + scoreName);
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
