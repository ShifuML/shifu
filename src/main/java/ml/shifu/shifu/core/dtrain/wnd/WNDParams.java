/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wnd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;

/**
 * {@link WNDParams} is message sent between master and workers for wide and deep model training.
 *
 * <p>
 * In worker, it will collect combined gradients and then send to master for merging. While in master when model weights
 * are updated, master will send model new weights to all works for next epoch iterations.
 * 
 * <p>
 * TODO define wide, dnn, embedding different weights/gradients parameters
 * TODO add model arch graph here
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WNDParams extends HaltBytable implements Combinable<WNDParams> {

    // TODO: add wide. dnn, embedding weights/gradients here
    
    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Combinable#combine(ml.shifu.guagua.io.Bytable)
     */
    @Override
    public WNDParams combine(WNDParams from) {
        // TODO How to combine workers into one to save memory
        return null;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.HaltBytable#doWrite(java.io.DataOutput)
     */
    @Override
    public void doWrite(DataOutput out) throws IOException {
        // TODO serialization
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.HaltBytable#doReadFields(java.io.DataInput)
     */
    @Override
    public void doReadFields(DataInput in) throws IOException {
        // TODO de-serialization
    }

}
