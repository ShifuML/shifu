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
import ml.shifu.shifu.container.ColumnScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.message.ColumnScoreMessage;
import ml.shifu.shifu.message.StatsResultMessage;
import ml.shifu.shifu.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;


/**
 * PostTrainWorker class calculate the average score for each bin of column
 */
public class PostTrainWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(PostTrainWorker.class);

    private List<ColumnScoreObject> colScoreList;
    private int receivedMsgCnt = 0;

    public PostTrainWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        colScoreList = new ArrayList<ColumnScoreObject>();
        receivedMsgCnt = 0;
    }

    /* (non-Javadoc)
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) {
        if (message instanceof ColumnScoreMessage) {
            ColumnScoreMessage msg = (ColumnScoreMessage) message;
            colScoreList.addAll(msg.getColScoreList());
            receivedMsgCnt++;

            log.debug("Received " + receivedMsgCnt + " messages, total message count is:" + msg.getTotalMsgCnt());

            if (receivedMsgCnt == msg.getTotalMsgCnt()) {
                // received all message, start to calculate

                int columnNum = msg.getColumnNum();
                ColumnConfig config = columnConfigList.get(columnNum);

                Double[] binScore = new Double[config.getBinLength()];
                Integer[] binCount = new Integer[config.getBinLength()];

                for (int i = 0; i < binScore.length; i++) {
                    binScore[i] = 0.0;
                    binCount[i] = 0;
                }

                for (ColumnScoreObject colScore : colScoreList) {
                    int binNum = CommonUtils.getBinNum(config, colScore.getColumnVal());
                    binScore[binNum] += Double.valueOf(colScore.getAvgScore());
                    binCount[binNum]++;
                }

                List<Integer> binAvgScore = new ArrayList<Integer>();
                for (int i = 0; i < binScore.length; i++) {
                    binScore[i] /= binCount[i];
                    binAvgScore.add((int) Math.round(binScore[i]));
                }

                config.setBinAvgScore(binAvgScore);
                nextActorRef.tell(new StatsResultMessage(config), getSelf());
            }
        } else {
            unhandled(message);
        }
    }

}
