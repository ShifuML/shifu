/**
 * Copyright [2012-2013] eBay Software Foundation
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
import com.google.inject.Injector;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.di.service.StatsService;
import ml.shifu.shifu.message.StatsResultMessage;
import ml.shifu.shifu.message.StatsValueObjectMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * StatsCalculateWorker class calculates the stats for each column
 * It will do the binning for the column, calculate max/min/average, and calculate KS/IV
 */
public class StatsCalculateWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(StatsCalculateWorker.class);
    private List<RawValueObject> rvoList;
    private int receivedMsgCnt;

    private StatsService statsService;


    public StatsCalculateWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            Injector injector,
            ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        rvoList = new ArrayList<RawValueObject>();
        receivedMsgCnt = 0;


        statsService = injector.getInstance(StatsService.class);

        Map<String, Object> params = new HashMap<String, Object>();

        params.put("numBins", modelConfig.getBinningExpectedNum());
        params.put("posTags", modelConfig.getPosTags());
        params.put("negTags", modelConfig.getNegTags());
        statsService.setParams(params);
    }

    @Override
    public void handleMsg(Object message) {
        if (message instanceof StatsValueObjectMessage) {
            log.debug("Received value object list for stats");
            StatsValueObjectMessage statsVoMessage = (StatsValueObjectMessage) message;
            receivedMsgCnt++;

            rvoList.addAll(statsVoMessage.getVoList());

            if (receivedMsgCnt == statsVoMessage.getTotalMsgCnt()) {
                log.debug("received " + receivedMsgCnt + ", start to work");
                ColumnConfig columnConfig = columnConfigList.get(statsVoMessage.getColumnNum());
                statsService.exec(columnConfig, rvoList);
                parentActorRef.tell(new StatsResultMessage(columnConfig), this.getSelf());
            }
        } else {
            unhandled(message);
        }

    }
}