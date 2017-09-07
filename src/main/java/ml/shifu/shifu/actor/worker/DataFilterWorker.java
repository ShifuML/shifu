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
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.message.NormPartRawDataMessage;
import ml.shifu.shifu.message.RunModelDataMessage;
import ml.shifu.shifu.message.StatsPartRawDataMessage;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;


/**
 * DataFilterWorker class is to filter data by setting.
 * It do filtering by the setting in @ModelConfig.dataSet.filterExpressions or
 */
public class DataFilterWorker extends AbstractWorkerActor {

    public static final Logger log = LoggerFactory.getLogger(DataFilterWorker.class);

    private DataPurifier dataPurifier;

    public DataFilterWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            ActorRef parentActorRef,
            ActorRef nextActorRef) throws IOException {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        dataPurifier = new DataPurifier(modelConfig);
    }

    public DataFilterWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            ActorRef parentActorRef,
            ActorRef nextActorRef,
            EvalConfig evalConfig) throws IOException {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        dataPurifier = new DataPurifier(evalConfig);
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.actor.worker.AbstractWorkerActor#handleMsg(java.lang.Object)

     */
    @Override
    public void handleMsg(Object message) throws Exception {
        if (message instanceof StatsPartRawDataMessage) {
            StatsPartRawDataMessage msg = (StatsPartRawDataMessage) message;
            purifyData(msg.getRawDataList());
            nextActorRef.tell(msg, getSelf());
        } else if (message instanceof NormPartRawDataMessage) {
            NormPartRawDataMessage msg = (NormPartRawDataMessage) message;
            purifyData(msg.getRawDataList());
            nextActorRef.tell(msg, getSelf());
        } else if (message instanceof RunModelDataMessage) {
            RunModelDataMessage msg = (RunModelDataMessage) message;
            purifyData(msg.getEvalDataList());
            nextActorRef.tell(msg, getSelf());
        } else {
            unhandled(message);
        }
    }

    /**
     * Filter the data - it uses @dataPurifier to filter data
     *
     * @param inputDataList - input data to filter
     */
    private void purifyData(List<String> inputDataList) {
        log.info("starting to filter data ... ");
        CollectionUtils.filter(inputDataList, new Predicate() {
            @Override
            public boolean evaluate(Object object) {
                String inputData = (String) object;
                return dataPurifier.isFilter(inputData);
            }
        });

        log.info("there are {} records after filter.", inputDataList.size());
    }

}
