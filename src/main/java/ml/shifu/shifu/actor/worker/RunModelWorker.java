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
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.core.model.ModelSpec;
import ml.shifu.shifu.message.RunModelDataMessage;
import ml.shifu.shifu.message.RunModelResultMessage;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.CollectionUtils;
import org.encog.ml.BasicML;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * RunModelWorker class computes the score for input data
 */
public class RunModelWorker extends AbstractWorkerActor {

    private ModelRunner modelRunner;

    public RunModelWorker(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, EvalConfig evalConfig,
            ActorRef parentActorRef, ActorRef nextActorRef) throws IOException {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, evalConfig, SourceType.LOCAL);

        String[] header = null;
        String delimiter = null;

        if( null == evalConfig
                || null == evalConfig.getDataSet().getHeaderPath()
                || null == evalConfig.getDataSet().getHeaderDelimiter()) {
            header = CommonUtils.getFinalHeaders(modelConfig);
            delimiter = modelConfig.getDataSetDelimiter();
        } else {
            header = CommonUtils.getFinalHeaders(evalConfig);
            delimiter = evalConfig.getDataSet().getDataDelimiter();
        }

        modelRunner = new ModelRunner(modelConfig, columnConfigList, header, delimiter, models);

        @SuppressWarnings("deprecation")
        boolean gbtConvertToProp = ((evalConfig == null) ? false :  evalConfig.getGbtConvertToProb());
        SourceType sourceType = ((evalConfig == null) ?
                modelConfig.getDataSet().getSource() : evalConfig.getDataSet().getSource());
        List<ModelSpec> subModels = CommonUtils.loadSubModels(modelConfig, this.columnConfigList, evalConfig,
                sourceType, gbtConvertToProp);
        if(CollectionUtils.isNotEmpty(subModels)) {
            for(ModelSpec modelSpec: subModels) {
                this.modelRunner.addSubModels(modelSpec);
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void handleMsg(Object message) {
        if(message instanceof RunModelDataMessage) {
            RunModelDataMessage msg = (RunModelDataMessage) message;
            List<String> evalDataList = msg.getEvalDataList();

            List<CaseScoreResult> scoreDataList = new ArrayList<CaseScoreResult>(evalDataList.size());
            for(String evalData: evalDataList) {
                CaseScoreResult scoreData = calculateModelScore(evalData);
                if(scoreData != null) {
                    scoreData.setInputData(evalData);
                    scoreDataList.add(scoreData);
                }
            }

            nextActorRef.tell(
                    new RunModelResultMessage(msg.getStreamId(), msg.getTotalStreamCnt(), msg.getMsgId(), msg
                            .isLastMsg(), scoreDataList), getSelf());
        } else {
            unhandled(message);
        }
    }

    /**
     * Call model runner to compute result score
     * 
     * @param evalData
     *            - data to run model
     * @return - the score result
     */
    private CaseScoreResult calculateModelScore(String evalData) {
        return modelRunner.compute(evalData);
    }

}
