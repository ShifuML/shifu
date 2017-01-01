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
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.message.StatsPartRawDataMessage;
import ml.shifu.shifu.message.TrainInstanceMessage;
import ml.shifu.shifu.message.TrainPartDataMessage;
import ml.shifu.shifu.util.Constants;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.buffer.BufferedMLDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * TrainDataPrepWorker class prepare the data for trainer
 * Notice: if the training data is too large, user can train model using disk
 */
public class TrainDataPrepWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(TrainModelWorker.class);

    private MLDataSet masterDataSet;
    private int receivedMsgCnt = 0;
    private List<AbstractTrainer> trainers;
    private boolean initialized = false;

    public TrainDataPrepWorker(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, ActorRef parentActorRef,
            ActorRef nextActorRef, List<AbstractTrainer> trainers) throws IOException {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        this.trainers = trainers;

        if(modelConfig.isTrainOnDisk()) {
            log.info("Training Option: Disk");
            ShifuFileUtils.createDirIfNotExists(Constants.TMP, SourceType.LOCAL);
            masterDataSet = new BufferedMLDataSet(new File(Constants.TMP, "master.egb"));
        } else {
            log.info("Training Option: Memory");
            masterDataSet = new BasicMLDataSet();
        }

    }

    /*
     * (non-Javadoc)
     * 
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) throws IOException {
        if(message instanceof TrainPartDataMessage) {
            log.debug("Received value object list for training model.");
            TrainPartDataMessage msg = (TrainPartDataMessage) message;
            for(MLDataPair mlDataPir: msg.getMlDataPairList()) {
                if(modelConfig.isTrainOnDisk() && !initialized) {
                    int inputSize = mlDataPir.getInput().size();
                    int idealSize = mlDataPir.getIdeal().size();
                    ((BufferedMLDataSet) masterDataSet).beginLoad(inputSize, idealSize);
                    initialized = true;
                }

                masterDataSet.add(mlDataPir);
            }
            receivedMsgCnt++;

            log.debug("Expected " + msg.getTotalMsgCnt() + " messages, received " + receivedMsgCnt + " message(s).");
            if(receivedMsgCnt == msg.getTotalMsgCnt()) {
                if(modelConfig.isTrainOnDisk() && initialized) {
                    ((BufferedMLDataSet) masterDataSet).endLoad();
                }

                for(AbstractTrainer trainer: trainers) {
                    // if the trainOnDisk is true, setting the "D" option
                    if(modelConfig.isTrainOnDisk()) {
                        trainer.setTrainingOption("D");
                    }

                    trainer.setDataSet(masterDataSet);
                    nextActorRef.tell(new TrainInstanceMessage(trainer), this.getSelf());
                }

                if(modelConfig.isTrainOnDisk() && initialized) {
                    masterDataSet.close();
                    masterDataSet = null;
                }
            }
        } else if(message instanceof StatsPartRawDataMessage) {
            StatsPartRawDataMessage msg = (StatsPartRawDataMessage) message;
            receivedMsgCnt++;

            log.debug("Expected " + msg.getTotalMsgCnt() + " messages, received " + receivedMsgCnt + " message(s).");
            if(receivedMsgCnt == msg.getTotalMsgCnt()) {
                for(AbstractTrainer trainer: trainers) {
                    // ((DecisionTreeTrainer)trainer).setDataSet(rawInstanceList);
                    nextActorRef.tell(new TrainInstanceMessage(trainer), this.getSelf());
                }
            }
        } else {
            unhandled(message);
        }
    }

}
