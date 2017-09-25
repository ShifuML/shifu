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

import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import akka.actor.UntypedActor;
import ml.shifu.shifu.util.CommonUtils;

/**
 * AbstractActor class
 * Abstract Actor is parent class of all kinds of work actor. It will contains the @ModelConfig and
 * 
 * <p>
 * ColumnConfig list for its sub-class, it also try to find the column number of the target column
 */
public abstract class AbstractActor extends UntypedActor {

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;
    protected AkkaExecStatus akkaStatus;

    protected int targetColumnNum = -1;
    protected boolean hasCandidates;

    public AbstractActor(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            AkkaExecStatus akkaStatus) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.akkaStatus = akkaStatus;

        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                targetColumnNum = config.getColumnNum();
                break;
            }
        }

        if(targetColumnNum == -1) {
            throw new RuntimeException("No Valid Target.");
        }

        this.hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
    }

    /**
     * If exception occurs in ActorSystem execution, set the AKKA status to false.
     * And put the exception into @AkkaExecStatus, this is for result checking
     * 
     * @param exception
     *            - exception occurred in execution
     */
    public void addExceptionIntoCondition(Exception exception) {
        akkaStatus.setStatus(false);
        akkaStatus.getCauses().add("Exception Occured when AKKA executing!");
        akkaStatus.setException(exception);
    }
}
