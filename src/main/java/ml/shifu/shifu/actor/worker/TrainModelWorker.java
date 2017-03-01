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
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.message.TrainInstanceMessage;
import ml.shifu.shifu.message.TrainResultMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

/**
 * TrainModelWorker class trains the models
 */
public class TrainModelWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(TrainModelWorker.class);

    public TrainModelWorker(
            ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList,
            ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
    }

    /* (non-Javadoc)
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) throws IOException {
        if (message instanceof TrainInstanceMessage) {
            log.info("Received train data for model training");
            TrainInstanceMessage msg = (TrainInstanceMessage) message;
            msg.getTrainer().train();

            nextActorRef.tell(new TrainResultMessage(), getSelf());
        } else {
            unhandled(message);
        }
    }
}
