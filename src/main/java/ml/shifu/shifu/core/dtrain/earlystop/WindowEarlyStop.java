package ml.shifu.shifu.core.dtrain.earlystop;

import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

public class WindowEarlyStop extends AbstractEarlyStopStrategy {
    private static final Logger LOGGER = LoggerFactory.getLogger(WindowEarlyStop.class);

    private double globalMinimumError = Double.MAX_VALUE;
    private double ignoreValue;
    private int minimumEpochs;
    private int bufferSize = 0;
    private int windowSize;

    @SuppressWarnings("rawtypes")
    public WindowEarlyStop(MasterContext context, ModelConfig modelConfig, int windowSize) {
        double minimumStepsRatio = DTrainUtils.getDouble(context.getProps(), // get # of steps to choose parameters
                CommonConstants.SHIFU_TRAIN_VAL_STEPS_RATIO, 0.01);
        this.ignoreValue = DTrainUtils.getDouble(context.getProps(), CommonConstants.EARLY_STOP_IGNORE_VALUE, 1.0E-20);
        this.minimumEpochs = (int)(modelConfig.getNumTrainEpochs() * minimumStepsRatio);
        this.windowSize = windowSize;
        LOGGER.info("Init WindowEarlyStop with ignore value {}, mini num epoch {}, windowSize {}", this.ignoreValue,
                this.minimumEpochs, this.windowSize);
    }

    @Override public boolean shouldEarlyStop(int epochs, double[] weights, double trainingError, double validationError) {
        if ( epochs < minimumEpochs ) {
            return false;
        }

        if ( validationError < this.globalMinimumError || validationError - this.globalMinimumError < ignoreValue) {
            this.globalMinimumError = validationError;
            this.bufferSize = 0;
            return false;
        } else {
            if ( this.bufferSize >= this.windowSize ) {
                return true;
            } else {
                this.bufferSize ++;
                return false;
            }
        }
    }
}
