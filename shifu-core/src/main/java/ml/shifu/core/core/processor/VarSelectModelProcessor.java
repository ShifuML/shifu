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
package ml.shifu.core.core.processor;

import ml.shifu.core.core.AbstractTrainer;
import ml.shifu.core.core.VariableSelector;
import ml.shifu.core.core.alg.NNTrainer;
import ml.shifu.core.core.validator.ModelInspector.ModelStep;
import ml.shifu.core.exception.ShifuErrorCode;
import ml.shifu.core.exception.ShifuException;
import ml.shifu.core.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Variable selection processor, select the variable based on KS/IV value, or </p>
 * <p/>
 * Selection variable based on the wrapper training processor. </p>
 */
public class VarSelectModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(VarSelectModelProcessor.class);

    /**
     * run for the variable selection
     */
    @Override
    public int run() throws Exception {
        setUp(ModelStep.VARSELECT);

        CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);

        VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);

        // Filter
        this.columnConfigList = selector.selectByFilter();
        try {
            this.saveColumnConfigList();
        } catch (ShifuException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
        }

        // Wrapper, only if enabled
        if (modelConfig.getVarSelectWrapperEnabled()) {
            wrapper(selector);
        }
        log.info("Step Finished: varselect");

        clearUp(ModelStep.VARSELECT);
        return 0;
    }

    /**
     * user wrapper to select variable
     *
     * @param selector
     * @throws Exception
     */
    private void wrapper(VariableSelector selector) throws Exception {

        NormalizeModelProcessor n = new NormalizeModelProcessor();

        //runNormalize();
        n.run();

        TrainModelProcessor t = new TrainModelProcessor(false, false);
        t.run();

        AbstractTrainer trainer = t.getTrainer(0);

        if (trainer instanceof NNTrainer) {
            selector.selectByWrapper((NNTrainer) trainer);
            try {
                this.saveColumnConfigList();
            } catch (ShifuException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
            }
        }
    }

}
