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

import ml.shifu.guagua.master.AbstractMasterComputable;
import ml.shifu.guagua.master.MasterContext;

/**
 * TODO master aggregation logic to aggregate gradients and update weights based on different optimization strategies
 * like ADAM, AdaGrad, SGD ...
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class WNDMaster extends AbstractMasterComputable<WNDParams, WNDParams> {

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.master.AbstractMasterComputable#init(ml.shifu.guagua.master.MasterContext)
     */
    @Override
    public void init(MasterContext<WNDParams, WNDParams> context) {
        // TODO initialize master parameters and load snapshot models
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.master.AbstractMasterComputable#doCompute(ml.shifu.guagua.master.MasterContext)
     */
    @Override
    public WNDParams doCompute(MasterContext<WNDParams, WNDParams> context) {
        // TODO aggregate gradients and according to optimization strategy to do weights update
        return null;
    }

}
