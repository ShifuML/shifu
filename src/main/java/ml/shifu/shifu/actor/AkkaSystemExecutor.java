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

import akka.actor.*;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.message.AkkaActorInputMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

/**
 * AkkaSystemExecutor class
 * The executor for AKKA system. It's singleton.
 */
public class AkkaSystemExecutor {

    private static Logger log = LoggerFactory.getLogger(AkkaSystemExecutor.class);
    private static AkkaSystemExecutor instance = new AkkaSystemExecutor();

    private ActorSystem actorSystem;

    // singleton
    private AkkaSystemExecutor() {
    }

    /**
     * Get executor for AKKA System
     * 
     * @return - executor
     */
    public static AkkaSystemExecutor getExecutor() {
        return instance;
    }

    /**
     * Submit job to calculate column stats
     * Column stats including:
     * - Binning of value range
     * - max/min/average
     * - ks/iv
     * 
     * @param modelConfig
     *            - configuration for model
     * @param columnConfigList
     *            - configurations for columns
     * @param scanners
     *            - scanners of training data
     */
    public void submitStatsCalJob(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            List<Scanner> scanners) {
        actorSystem = ActorSystem.create("ShifuActorSystem");
        final AkkaExecStatus akkaStatus = new AkkaExecStatus(true);

        log.info("Create Akka system to calculate stats");
        ActorRef statsCalRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -1437127862571741369L;

            public UntypedActor create() {
                return new CalculateStatsActor(modelConfig, columnConfigList, akkaStatus);
            }
        }), "stats-calculator");

        statsCalRef.tell(new AkkaActorInputMessage(scanners), statsCalRef);

        // wait for termination and check the status
        actorSystem.awaitTermination();
        checkAkkaStatus(akkaStatus);
    }

    /**
     * Submit job to normalize training data
     * 
     * @param modelConfig
     *            - configuration for model
     * @param columnConfigList
     *            - configurations for columns
     * @param scanners
     *            - scanners of training data
     */
    public void submitNormalizeJob(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            List<Scanner> scanners) {
        actorSystem = ActorSystem.create("ShifuActorSystem");
        final AkkaExecStatus akkaStatus = new AkkaExecStatus(true);

        log.info("Create Akka system to normalize data");
        ActorRef dataNormalizeRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -2123098236012879296L;

            public UntypedActor create() throws IOException {
                return new NormalizeDataActor(modelConfig, columnConfigList, akkaStatus);
            }
        }), "data-normalizer");

        dataNormalizeRef.tell(new AkkaActorInputMessage(scanners), dataNormalizeRef);

        // wait for termination
        actorSystem.awaitTermination();
        checkAkkaStatus(akkaStatus);
    }

    /**
     * Submit job to training model
     * 
     * @param modelConfig
     *            - configuration for model
     * @param columnConfigList
     *            - configurations for columns
     * @param scanners
     *            - scanners of normalized training data
     * @param trainers
     *            - model trainer
     */
    public void submitModelTrainJob(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            List<Scanner> scanners, final List<AbstractTrainer> trainers) {
        actorSystem = ActorSystem.create("ShifuActorSystem");
        final AkkaExecStatus akkaStatus = new AkkaExecStatus(true);

        log.info("Create Akka system to train model");
        ActorRef modelTrainerRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -1437127862571741369L;

            public UntypedActor create() {
                return new TrainModelActor(modelConfig, columnConfigList, akkaStatus, trainers);
            }
        }), "model-trainer");

        modelTrainerRef.tell(new AkkaActorInputMessage(scanners), modelTrainerRef);

        // wait for termination
        actorSystem.awaitTermination();
        checkAkkaStatus(akkaStatus);
    }

    /**
     * Submit job to training decision-tree model
     * 
     * @param modelConfig
     *            - configuration for model
     * @param columnConfigList
     *            - configurations for columns
     * @param scanners
     *            - scanners of normalized training data
     * @param trainers
     *            - model trainer
     */
    public void submitDecisionTreeTrainJob(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            List<Scanner> scanners, final List<AbstractTrainer> trainers) {
        actorSystem = ActorSystem.create("ShifuActorSystem");
        final AkkaExecStatus akkaStatus = new AkkaExecStatus(true);

        log.info("Create Akka system to train dt-model");
        ActorRef modelTrainerRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = 2394968604729416422L;

            public UntypedActor create() {
                return new TrainDtModelActor(modelConfig, columnConfigList, akkaStatus, trainers);
            }
        }), "dt-model-trainer");

        modelTrainerRef.tell(new AkkaActorInputMessage(scanners), modelTrainerRef);

        // wait for termination
        actorSystem.awaitTermination();
        checkAkkaStatus(akkaStatus);
    }

    /**
     * Submit job to post-train the model
     * 
     * @param modelConfig
     *            - configuration for model
     * @param columnConfigList
     *            - configurations for columns
     * @param scanners
     *            - scanners of select data that are normalized
     */
    public void submitPostTrainJob(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            List<Scanner> scanners) {
        actorSystem = ActorSystem.create("ShifuActorSystem");
        final AkkaExecStatus akkaStatus = new AkkaExecStatus(true);

        log.info("Create Akka system to post-train model");
        ActorRef postTrainerRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -1437127862571741369L;

            public UntypedActor create() {
                return new PostTrainActor(modelConfig, columnConfigList, akkaStatus);
            }
        }), "model-posttrainer");

        postTrainerRef.tell(new AkkaActorInputMessage(scanners), postTrainerRef);

        // wait for termination
        actorSystem.awaitTermination();
        checkAkkaStatus(akkaStatus);
    }

    /**
     * Submit job to run model evaluation
     * 
     * @param modelConfig
     *            - configuration for model
     * @param columnConfigList
     *            - configurations for columns
     * @param evalConfig
     *            the eval config instance
     * @param scanners
     *            - scanners of evaluation data
     */
    public void submitModelEvalJob(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            final EvalConfig evalConfig, List<Scanner> scanners) {
        actorSystem = ActorSystem.create("ShifuActorSystem");
        final AkkaExecStatus akkaStatus = new AkkaExecStatus(true);

        log.info("Create Akka system to evaluate model");
        ActorRef modelEvalRef = actorSystem.actorOf(new Props(new UntypedActorFactory() {
            private static final long serialVersionUID = -1437127862571741369L;

            public UntypedActor create() {
                return new EvalModelActor(modelConfig, columnConfigList, akkaStatus, evalConfig);
            }
        }), "model-evaluator");

        modelEvalRef.tell(new AkkaActorInputMessage(scanners), modelEvalRef);

        // wait for termination
        actorSystem.awaitTermination();
        checkAkkaStatus(akkaStatus);
    }

    /**
     * check the execute status of AKKA, if there is any Exceptions, wrap it with ShifuException and throw it
     * 
     * @param akkaStatus
     */
    private void checkAkkaStatus(final AkkaExecStatus akkaStatus) {
        if(!akkaStatus.getStatus()) {
            throw new ShifuException(ShifuErrorCode.ERROR_AKKA_EXECUTE_EXCEPTION, akkaStatus.getException());
        }
    }
}
