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
import ml.shifu.core.container.CaseScoreResult;
import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.container.obj.EvalConfig;
import ml.shifu.core.container.obj.ModelConfig;
import ml.shifu.core.container.obj.RawSourceData.SourceType;
import ml.shifu.core.core.ModelRunner;
import ml.shifu.core.message.RunModelDataMessage;
import ml.shifu.core.message.RunModelResultMessage;
import ml.shifu.core.util.CommonUtils;
import org.encog.ml.BasicML;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * RunModelWorker class computes the score for input data
 */
public class RunModelWorker extends AbstractWorkerActor {

    private ModelRunner modelRunner;

    /**
     * @param modelConfig
     * @param columnConfigList
     * @param parentActorRef
     * @param nextActorRef
     * @throws IOException
     */
    public RunModelWorker(ModelConfig modelConfig,
                          List<ColumnConfig> columnConfigList, EvalConfig evalConfig,
                          ActorRef parentActorRef, ActorRef nextActorRef) throws IOException {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, evalConfig, SourceType.LOCAL);

        String[] header = null;
        String delimiter = null;

        if (null == evalConfig
                || null == evalConfig.getDataSet().getHeaderPath()
                || null == evalConfig.getDataSet().getHeaderDelimiter()) {

            header = CommonUtils
                    .getHeaders(modelConfig.getDataSet().getHeaderPath(),
                            modelConfig.getDataSet().getHeaderDelimiter(),
                            modelConfig.getDataSet().getSource());

            delimiter = modelConfig.getDataSetDelimiter();

        } else {
            header = CommonUtils
                    .getHeaders(evalConfig.getDataSet().getHeaderPath(),
                            evalConfig.getDataSet().getHeaderDelimiter(),
                            evalConfig.getDataSet().getSource());

            delimiter = evalConfig.getDataSet().getDataDelimiter();
        }

        modelRunner = new ModelRunner(modelConfig, columnConfigList, header,
                delimiter, models);
    }

    /*
     * (non-Javadoc)
     *
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) {
        if (message instanceof RunModelDataMessage) {
            RunModelDataMessage msg = (RunModelDataMessage) message;
            List<String> evalDataList = msg.getEvalDataList();

            List<CaseScoreResult> scoreDataList = new ArrayList<CaseScoreResult>(evalDataList.size());
            for (String evalData : evalDataList) {
                CaseScoreResult scoreData = calculateModelScore(evalData);
                if (scoreData != null) {
                    scoreData.setInputData(evalData);
                    scoreDataList.add(scoreData);
                }
            }

            nextActorRef.tell(new RunModelResultMessage(msg.getStreamId(),
                    msg.getTotalStreamCnt(), msg.getMsgId(), msg.isLastMsg(), scoreDataList), getSelf());
        } else {
            unhandled(message);
        }
    }

    /**
     * Call model runner to compute result score
     *
     * @param evalData - data to run model
     * @return - the score result
     */
    private CaseScoreResult calculateModelScore(String evalData) {
        return modelRunner.compute(evalData);
    }

}
