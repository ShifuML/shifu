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

import akka.actor.UntypedActor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * AbstractActor class
 * Abstract Actor is parent class of all kinds of work actor. It will contains the @ModelConfig and
 * 
 * <p>
 * ColumnConfig list for its sub-class, it also try to find the column number of the target column
 */
public abstract class AbstractActor extends UntypedActor {
    private static Logger log = LoggerFactory.getLogger(AbstractActor.class);

    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;
    protected AkkaExecStatus akkaStatus;

    protected int targetColumnNum = -1;

    public AbstractActor(final ModelConfig modelConfig, final List<ColumnConfig> columnConfigList,
            AkkaExecStatus akkaStatus) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.akkaStatus = akkaStatus;

        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                targetColumnNum = config.getColumnNum();
                log.debug("Target Column Name: " + config.getColumnName());
                log.debug("Target Column Num: " + targetColumnNum);
                break;
            }
        }

        if(targetColumnNum == -1) {
            throw new RuntimeException("No Valid Target.");
        }
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
