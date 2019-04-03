package ml.shifu.shifu.core.dtrain.earlystop;

import ml.shifu.shifu.core.ConvergeJudger;
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
 **/

public class ConvergeAndValidToleranceEarlyStop extends AbstractEarlyStopStrategy {

    private static final Logger LOG = LoggerFactory.getLogger(ConvergeAndValidToleranceEarlyStop.class);

    /**
     * Convergence judger instance for convergence checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();

    /**
     * Convergence threshold setting.
     */
    private double convergenceThreshold;

    /**
     * Validation tolerance which is for early stop, by default it is 0d which means early stop is not enabled.
     */
    private double validationTolerance;

    /**
     * The weights for previous
     */
    private double[] previousWeights;

    public ConvergeAndValidToleranceEarlyStop(double convergenceThreshold, double validationTolerance) {
        this.convergenceThreshold = convergenceThreshold;
        this.validationTolerance = validationTolerance;
    }

    @Override public boolean shouldEarlyStop(int epochs, double[] weights, double trainingError,
            double validationError) {
        boolean vtTriggered = false;

        if(validationTolerance > 0d && this.previousWeights != null) {
            double weightSumSquare = 0d;
            double diffWeightSumSquare = 0d;
            for(int i = 0; i < weights.length; i++) {
                weightSumSquare += Math.pow(weights[i], 2);
                diffWeightSumSquare += Math.pow(weights[i] - this.previousWeights[i], 2);
            }
            if(Math.pow(diffWeightSumSquare, 0.5) < this.validationTolerance * Math
                    .max(Math.pow(weightSumSquare, 0.5), 1d)) {
                LOG.info("Debug: diffWeightSumSquare {}, weightSumSquare {}, validationTolerance {}",
                        Math.pow(diffWeightSumSquare, 0.5), Math.pow(weightSumSquare, 0.5), validationTolerance);
                vtTriggered = true;
            }
        }

        this.previousWeights = weights;

        // Convergence judging part
        double avgErr = (trainingError + validationError) / 2;

        if(judger.judge(avgErr, convergenceThreshold) || vtTriggered) {
            LOG.info("NNMaster compute iteration {} converged !", epochs);
            return true;
        } else {
            LOG.debug("NNMaster compute iteration {} not converged yet !", epochs);
            return false;
        }
    }
}
