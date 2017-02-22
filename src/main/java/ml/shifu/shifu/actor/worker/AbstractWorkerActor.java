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
import ml.shifu.shifu.actor.AbstractActor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.message.ExceptionMessage;

import java.util.List;

/**
 * AbstractWorkerActor class is the abstract class for all worker actor
 * Each work actor contains its parent actor and next actor, so that it can send result
 * to where. and it can also send message to parent directly, when exception happened.
 * <p>
 * Notice, if the worker actor is the last step of whole loop, its next actor will be the same as its parent actor
 */
public abstract class AbstractWorkerActor extends AbstractActor {

    protected ActorRef parentActorRef;
    protected ActorRef nextActorRef;

    public AbstractWorkerActor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, null);
        this.parentActorRef = parentActorRef;
        this.nextActorRef = nextActorRef;
    }

    /*
     * (non-Javadoc)
     * 
     * @see akka.actor.UntypedActor#onReceive(java.lang.Object)
     */
    @Override
    public void onReceive(Object message) {
        try {
            handleMsg(message);
        } catch (Exception e) {
            parentActorRef.tell(new ExceptionMessage(e), getSelf());
        }
    }

    /**
     * The method to handle message
     * 
     * @param message
     *            - received message
     * 
     * @throws Exception
     *             if any exception
     */
    public abstract void handleMsg(Object message) throws Exception;
}
